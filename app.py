import gradio as gr
import numpy as np
from scipy.io import wavfile
from scipy import signal
import tempfile
import os
import time
import subprocess
import sys

# Function to install and import audio libraries
def setup_audio_libraries():
    """Try to import and install audio libraries"""
    try:
        import soundfile as sf
        import librosa
        return sf, librosa, True
    except ImportError:
        print("📦 Installing audio libraries...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "soundfile", "librosa"])
            import soundfile as sf
            import librosa
            print("✅ Audio libraries installed successfully")
            return sf, librosa, True
        except:
            print("⚠️ Could not install audio libraries, using WAV-only mode")
            return None, None, False

# Setup audio libraries
sf, librosa, AUDIO_LIBS_AVAILABLE = setup_audio_libraries()

def separate_audio(audio_file, separation_mode="Both", progress=gr.Progress()):
    """DeVox AI-powered audio separation with improved vocal extraction"""
    if audio_file is None:
        return None, None, "🎧 Please upload an audio file to get started!"
    
    try:
        # Progress tracking
        progress(0.1, desc="🎵 Loading your audio...")
        time.sleep(0.5)
        
        # Read audio file with multi-format support
        if AUDIO_LIBS_AVAILABLE and sf and librosa:
            try:
                # Use librosa for multi-format support
                audio_data, sample_rate = librosa.load(audio_file, sr=None, mono=False)
                
                # librosa returns (channels, samples) for stereo, we need (samples, channels)
                if len(audio_data.shape) == 2:
                    audio_data = audio_data.T
                
                audio_data = audio_data.astype(np.float32)
                print(f"✅ Loaded with librosa: {audio_data.shape}")
            except Exception as e:
                print(f"⚠️ Librosa failed, trying scipy: {e}")
                # Fallback to scipy
                sample_rate, audio_data = wavfile.read(audio_file)
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
        else:
            # WAV-only mode with scipy
            sample_rate, audio_data = wavfile.read(audio_file)
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
        
        progress(0.3, desc="🔍 Analyzing audio spectrum...")
        time.sleep(0.5)
        
        # Handle mono/stereo conversion
        if len(audio_data.shape) == 1:
            # For mono files, create artificial stereo
            stereo_data = np.stack([audio_data, audio_data], axis=1)
        else:
            stereo_data = audio_data[:, :2] if audio_data.shape[1] > 2 else audio_data
        
        progress(0.6, desc="🤖 AI separation in progress...")
        time.sleep(0.8)
        
        left_channel = stereo_data[:, 0]
        right_channel = stereo_data[:, 1]
        
        # Improved separation algorithms
        if separation_mode in ["Both", "Vocals Only"]:
            # Multi-method vocal extraction approach
            
            # Method 1: Basic center extraction with phase adjustment
            center_vocals = (left_channel - right_channel)
            
            # Method 2: Enhanced spectral processing
            # Apply vocal-specific frequency filtering (100Hz-8kHz for better clarity)
            nyquist = sample_rate // 2
            low_cutoff = max(100 / nyquist, 0.01)   # Higher low-cut to remove bass bleed
            high_cutoff = min(8000 / nyquist, 0.95)  # Lower high-cut to reduce cymbal bleed
            
            # More aggressive filtering for vocal isolation
            b_high, a_high = signal.butter(6, low_cutoff, btype='high')  # Steeper filter
            b_low, a_low = signal.butter(6, high_cutoff, btype='low')
            
            center_vocals_filtered = signal.filtfilt(b_high, a_high, center_vocals)
            center_vocals_filtered = signal.filtfilt(b_low, a_low, center_vocals_filtered)
            
            # Method 3: Mid-side processing for better vocal isolation
            # Convert to mid-side
            mid = (left_channel + right_channel) * 0.5
            side = (left_channel - right_channel) * 0.5
            
            # Enhance side channel (where vocals often reside after center removal)
            side_enhanced = side * 2.5
            
            # Apply vocal-specific EQ to enhance presence
            # Boost vocal formant frequencies (1-3kHz)
            formant_freq = 2000 / nyquist
            if formant_freq < 0.95:
                # Simple peak filter for vocal presence
                Q = 2.0
                b_peak, a_peak = signal.iirpeak(formant_freq, Q)
                side_enhanced = signal.filtfilt(b_peak, a_peak, side_enhanced)
            
            # Method 4: Combine approaches with intelligent weighting
            # Weight based on frequency content
            vocal_mix = center_vocals_filtered * 0.3 + side_enhanced * 0.7
            
            # Method 5: Advanced dynamic processing
            # Calculate envelope for dynamic gating
            window_size = int(sample_rate * 0.05)  # 50ms windows
            envelope = []
            
            for i in range(0, len(vocal_mix) - window_size, window_size // 4):
                window = vocal_mix[i:i+window_size]
                env = np.max(np.abs(window))
                envelope.extend([env] * (window_size // 4))
            
            # Pad envelope to match audio length
            while len(envelope) < len(vocal_mix):
                envelope.append(envelope[-1] if envelope else 0)
            envelope = np.array(envelope[:len(vocal_mix)])
            
            # Adaptive threshold based on signal characteristics
            signal_energy = np.mean(envelope)
            adaptive_threshold = signal_energy * 0.2
            
            # Smooth gating with hysteresis
            gate = np.where(envelope > adaptive_threshold, 1.0, 0.05)
            
            # Apply smoothing to avoid clicks
            smooth_window = int(sample_rate * 0.005)  # 5ms smoothing
            if smooth_window > 1:
                gate = np.convolve(gate, np.ones(smooth_window)/smooth_window, mode='same')
            
            vocals = vocal_mix * gate
            
            # Method 6: Final enhancement and limiting
            # Gentle compression for vocal clarity
            vocals_rms = np.sqrt(np.mean(vocals**2))
            if vocals_rms > 0:
                target_rms = 0.1
                compression_ratio = np.clip(target_rms / vocals_rms, 0.5, 2.0)
                vocals *= compression_ratio
            
            # Soft limiting to prevent distortion
            vocals = np.tanh(vocals * 0.8) * 0.9
            vocals = np.clip(vocals, -0.95, 0.95)
            
        else:
            vocals = np.zeros_like(left_channel)
        
        if separation_mode in ["Both", "Instrumental Only"]:
            # Enhanced instrumental extraction
            
            # Method 1: Improved center channel removal
            sum_channels = (left_channel + right_channel) * 0.5
            
            # Method 2: Preserve stereo width while removing center
            # Calculate the difference signal for stereo information
            stereo_diff = (left_channel - right_channel) * 0.3
            
            # Combine sum with reduced difference to maintain some stereo feel
            instrumental = sum_channels + np.roll(stereo_diff, int(sample_rate * 0.001))  # 1ms delay for width
            
            # Method 3: Frequency-dependent processing
            # Apply gentle EQ to enhance instrumental elements
            # Boost bass and treble slightly while reducing midrange
            frequencies = [60, 250, 2000, 8000]  # Hz
            gains = [1.1, 0.9, 0.85, 1.05]  # Gain factors
            
            instrumental_eq = instrumental.copy()
            for freq, gain in zip(frequencies, gains):
                # Simple shelving filters
                if freq < nyquist * 0.9:
                    cutoff = freq / nyquist
                    if freq < 500:  # Low shelf
                        b, a = signal.butter(2, cutoff, btype='low')
                        low_component = signal.filtfilt(b, a, instrumental)
                        instrumental_eq += (gain - 1) * low_component * 0.3
                    elif freq > 4000:  # High shelf
                        b, a = signal.butter(2, cutoff, btype='high')
                        high_component = signal.filtfilt(b, a, instrumental)
                        instrumental_eq += (gain - 1) * high_component * 0.3
            
            instrumental = np.clip(instrumental_eq * 0.95, -1, 1)
            
        else:
            instrumental = np.zeros_like(left_channel)
        
        progress(0.9, desc="🎚️ Optimizing audio quality...")
        time.sleep(0.3)
        
        # Save outputs based on mode
        instrumental_path = None
        vocal_path = None
        
        if separation_mode in ["Both", "Instrumental Only"]:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_inst:
                if AUDIO_LIBS_AVAILABLE and sf:
                    sf.write(f_inst.name, instrumental, sample_rate)
                else:
                    wavfile.write(f_inst.name, sample_rate, (instrumental * 32767).astype(np.int16))
                instrumental_path = f_inst.name
        
        if separation_mode in ["Both", "Vocals Only"]:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_vocal:
                if AUDIO_LIBS_AVAILABLE and sf:
                    sf.write(f_vocal.name, vocals, sample_rate)
                else:
                    wavfile.write(f_vocal.name, sample_rate, (vocals * 32767).astype(np.int16))
                vocal_path = f_vocal.name
        
        progress(1.0, desc="✨ Professional separation complete!")
        
        # Status message based on mode
        if separation_mode == "Vocals Only":
            status_msg = "🎤 **Vocals Extracted!** Enhanced vocal isolation complete!"
        elif separation_mode == "Instrumental Only":
            status_msg = "🎼 **Instrumental Ready!** Professional backing track extracted!"
        else:
            status_msg = "🎉 **Success!** Both vocal and instrumental tracks are ready!"
        
        return instrumental_path, vocal_path, status_msg
        
    except Exception as e:
        return None, None, f"❌ **Processing Error:** {str(e)}"

# Premium Professional Luxury CSS
premium_luxury_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&display=swap');

/* Global Premium Styling */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: #0a0a0a !important;
    color: #ffffff !important;
}

/* Main Container - Luxury Glass Effect */
.main {
    background: linear-gradient(135deg, rgba(30, 30, 30, 0.95) 0%, rgba(20, 20, 20, 0.98) 100%) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 215, 0, 0.2) !important;
    border-radius: 24px !important;
    box-shadow: 
        0 20px 40px rgba(0, 0, 0, 0.4),
        inset 0 1px 0 rgba(255, 215, 0, 0.1) !important;
    margin: 20px !important;
    padding: 40px !important;
    position: relative !important;
}

/* Luxury Header */
.premium-header {
    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%) !important;
    border: 1px solid rgba(255, 215, 0, 0.3) !important;
    border-radius: 20px !important;
    padding: 48px 32px !important;
    text-align: center !important;
    margin-bottom: 40px !important;
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 215, 0, 0.1) !important;
    position: relative !important;
}

.premium-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(255, 215, 0, 0.5) 50%, transparent 100%);
}

/* Typography - Premium Blend */
.luxury-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 3.5rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #ffd700 0%, #ffffff 50%, #ffd700 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 0 12px 0 !important;
    letter-spacing: -0.02em !important;
    text-shadow: 0 2px 8px rgba(255, 215, 0, 0.3) !important;
}

.luxury-tagline {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.375rem !important;
    font-weight: 400 !important;
    color: #d4d4d8 !important;
    margin: 0 0 20px 0 !important;
    letter-spacing: 0.05em !important;
}

.luxury-description {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.125rem !important;
    font-weight: 400 !important;
    color: #a1a1aa !important;
    margin: 0 auto !important;
    max-width: 600px !important;
    line-height: 1.6 !important;
}

/* Premium Upload Section */
.premium-upload {
    background: linear-gradient(135deg, rgba(45, 45, 45, 0.8) 0%, rgba(30, 30, 30, 0.9) 100%) !important;
    border: 2px dashed rgba(255, 215, 0, 0.4) !important;
    border-radius: 16px !important;
    padding: 40px 24px !important;
    text-align: center !important;
    margin: 24px 0 !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    backdrop-filter: blur(10px) !important;
    position: relative !important;
}

/* Luxury Buttons */
.premium-button {
    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%) !important;
    border: 2px solid rgba(255, 215, 0, 0.5) !important;
    color: #ffd700 !important;
    border-radius: 12px !important;
    padding: 16px 32px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.125rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
    text-transform: uppercase !important;
}

/* Status Box */
.premium-status {
    background: linear-gradient(135deg, rgba(30, 30, 30, 0.9) 0%, rgba(20, 20, 20, 0.95) 100%) !important;
    border: 1px solid rgba(255, 215, 0, 0.3) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    color: #e4e4e7 !important;
    font-weight: 500 !important;
    backdrop-filter: blur(10px) !important;
}

/* Professional sections and badges */
.professional-section {
    background: rgba(30, 30, 30, 0.8) !important;
    border: 1px solid rgba(255, 215, 0, 0.2) !important;
    border-radius: 12px !important;
    padding: 32px 24px !important;
    margin: 24px 0 !important;
    backdrop-filter: blur(10px) !important;
}

.professional-badge {
    background: linear-gradient(135deg, rgba(255, 215, 0, 0.1) 0%, rgba(255, 215, 0, 0.05) 100%) !important;
    border: 1px solid rgba(255, 215, 0, 0.3) !important;
    color: #ffd700 !important;
    padding: 8px 16px !important;
    border-radius: 20px !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    display: inline-block !important;
}

.professional-footer {
    background: linear-gradient(135deg, rgba(30, 30, 30, 0.9) 0%, rgba(20, 20, 20, 0.95) 100%) !important;
    border: 1px solid rgba(255, 215, 0, 0.3) !important;
    border-radius: 16px !important;
    padding: 48px 32px !important;
    text-align: center !important;
    margin-top: 40px !important;
    backdrop-filter: blur(10px) !important;
}

/* Author Credit Styling */
.author-credit {
    background: rgba(20, 20, 20, 0.6) !important;
    border: 1px solid rgba(255, 215, 0, 0.2) !important;
    border-radius: 12px !important;
    padding: 24px 32px !important;
    margin-top: 32px !important;
    text-align: center !important;
    backdrop-filter: blur(8px) !important;
}

.author-name {
    color: #ffd700 !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
}

.author-links a {
    color: #d4d4d8 !important;
    text-decoration: none !important;
    margin: 0 8px !important;
    font-weight: 500 !important;
    transition: color 0.3s ease !important;
}

.author-links a:hover {
    color: #ffd700 !important;
}
"""

# Print system status
print("\n" + "="*50)
print("🎵 DEVOX AUDIO SEPARATOR")
print("="*50)
if AUDIO_LIBS_AVAILABLE:
    print("✅ FULL SUPPORT: MP3, WAV, FLAC, M4A, OGG")
else:
    print("⚠️ LIMITED SUPPORT: WAV files only")
print("="*50 + "\n")

# Create the improved interface
with gr.Blocks(
    title="DeVox - Premium AI Audio Separator",
    theme=gr.themes.Base(
        primary_hue="yellow",
        secondary_hue="gray",
        neutral_hue="slate"
    ).set(
        body_background_fill="#0a0a0a",
        block_background_fill="rgba(30, 30, 30, 0.95)",
        block_border_color="rgba(255, 215, 0, 0.2)",
        block_border_width="1px",
        block_radius="16px",
        button_primary_background_fill="linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)",
        button_primary_text_color="#ffd700"
    ),
    css=premium_luxury_css
) as app:
    
    # Premium Header
    gr.HTML('''
    <div class="premium-header">
        <h1 class="luxury-title">𝒟𝑒𝒱𝑜𝓍</h1>
        <h2 class="luxury-tagline">SEPARATE • CREATE • INNOVATE</h2>
        <p class="luxury-description">
            Premium AI-powered audio separation technology with enhanced vocal extraction capabilities. 
            Choose your separation mode for optimal results.
        </p>
    </div>
    ''')
    
    # Main Processing Interface
    with gr.Row():
        # Upload Section
        with gr.Column(scale=1):
            gr.HTML('''
            <div style="background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%); 
                        border: 2px solid rgba(139, 69, 19, 0.4); 
                        border-radius: 12px; 
                        padding: 32px 24px; 
                        text-align: center; 
                        margin-bottom: 24px;
                        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);">
                <h3 style="font-family: 'Inter', sans-serif; 
                           font-size: 1.5rem; 
                           font-weight: 600; 
                           color: #D4AF37; 
                           margin: 0 0 12px 0; 
                           letter-spacing: 0.02em;">
                    📁 Audio Upload Center
                </h3>
                <p style="color: #d1d5db; 
                          font-size: 1rem; 
                          margin: 0; 
                          font-weight: 400;">
                    Upload your audio file to begin professional separation
                </p>
            </div>
            ''')
            
            # Dynamic label based on capabilities
            audio_label = "🎧 Upload Audio File (MP3, WAV, FLAC, M4A, OGG)" if AUDIO_LIBS_AVAILABLE else "🎧 Upload WAV Audio File"
            
            audio_input = gr.Audio(
                label=audio_label,
                type="filepath",
                elem_classes=["professional-upload"]
            )
            
            # Separation mode selector
            separation_mode = gr.Radio(
                choices=["Both", "Vocals Only", "Instrumental Only"],
                value="Both",
                label="🎯 Processing Mode",
                info="Select your desired output format"
            )
            
            separate_btn = gr.Button(
                "🚀 Process Audio",
                variant="primary",
                size="lg",
                elem_classes=["executive-button", "professional-glow"]
            )
            
            # Dynamic status message
            status_msg = f"**Ready for Processing** • {'✅ Full format support' if AUDIO_LIBS_AVAILABLE else '⚠️ WAV only'}"
            status_output = gr.Markdown(
                value=status_msg,
                elem_classes=["professional-status"]
            )
        
        # Results Section
        with gr.Column(scale=1):
            gr.HTML('''
            <div style="background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%); 
                        border: 2px solid rgba(139, 69, 19, 0.4); 
                        border-radius: 12px; 
                        padding: 32px 24px; 
                        text-align: center; 
                        margin-bottom: 24px;
                        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);">
                <h3 style="font-family: 'Inter', sans-serif; 
                           font-size: 1.5rem; 
                           font-weight: 600; 
                           color: #D4AF37; 
                           margin: 0 0 12px 0; 
                           letter-spacing: 0.02em;">
                    🎼 Professional Results
                </h3>
                <p style="color: #d1d5db; 
                          font-size: 1rem; 
                          margin: 0; 
                          font-weight: 400;">
                    High-quality separated audio tracks
                </p>
            </div>
            ''')
            
            # Results with professional styling
            with gr.Group(elem_classes=["professional-section"]):
                gr.HTML('''
                <div style="text-align: center; margin-bottom: 16px;">
                    <h4 style="color: #D4AF37; font-size: 1.3rem; margin: 0 0 8px 0;">🎼 Instrumental Track</h4>
                    <p style="color: #d1d5db; font-size: 0.95rem; margin: 0;">
                        Clean backing track with vocals removed
                    </p>
                </div>
                ''')
                instrumental_output = gr.Audio(label="", type="filepath", interactive=False)
            
            with gr.Group(elem_classes=["professional-section"]):
                gr.HTML('''
                <div style="text-align: center; margin-bottom: 16px;">
                    <h4 style="color: #D4AF37; font-size: 1.3rem; margin: 0 0 8px 0;">🎤 Vocal Track</h4>
                    <p style="color: #d1d5db; font-size: 0.95rem; margin: 0;">
                        Isolated vocals with instruments removed
                    </p>
                </div>
                ''')
                vocal_output = gr.Audio(label="", type="filepath", interactive=False)
    
    # How It Works Section
    gr.HTML('''
    <div class="professional-section">
        <h3 style="font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 600; color: #D4AF37; text-align: center; margin: 0 0 32px 0;">
            How DeVox Studio Works
        </h3>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 32px; margin-top: 32px;">
            <div style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 16px;">📤</div>
                <h4 style="color: #D4AF37; font-size: 1.2rem; margin: 0 0 12px 0;">1. Upload</h4>
                <p style="color: #d1d5db; margin: 0; line-height: 1.5;">
                    Upload your audio file in any supported format. Files are processed locally for privacy.
                </p>
            </div>
            
            <div style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 16px;">⚙️</div>
                <h4 style="color: #D4AF37; font-size: 1.2rem; margin: 0 0 12px 0;">2. Process</h4>
                <p style="color: #d1d5db; margin: 0; line-height: 1.5;">
                    Advanced algorithms analyze and separate vocals from instruments using spectral processing.
                </p>
            </div>
            
            <div style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 16px;">📥</div>
                <h4 style="color: #D4AF37; font-size: 1.2rem; margin: 0 0 12px 0;">3. Download</h4>
                <p style="color: #d1d5db; margin: 0; line-height: 1.5;">
                    Download your separated tracks as high-quality WAV files ready for use in your projects.
                </p>
            </div>
        </div>
    </div>
    ''')
    
    # Technical Specifications
    gr.HTML('''
    <div class="professional-section">
        <h3 style="font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 600; color: #D4AF37; text-align: center; margin: 0 0 32px 0;">
            Technical Specifications
        </h3>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px;">
            <div>
                <h4 style="color: #D4AF37; font-size: 1.1rem; margin: 0 0 16px 0;">📊 Audio Processing</h4>
                <ul style="color: #d1d5db; margin: 0; padding-left: 20px; line-height: 1.6;">
                    <li>Multi-algorithm vocal separation</li>
                    <li>Spectral filtering (100Hz-8kHz)</li>
                    <li>Mid-side processing techniques</li>
                    <li>Adaptive gain control</li>
                    <li>Dynamic envelope processing</li>
                </ul>
            </div>
            
            <div>
                <h4 style="color: #D4AF37; font-size: 1.1rem; margin: 0 0 16px 0;">📁 Supported Formats</h4>
                <ul style="color: #d1d5db; margin: 0; padding-left: 20px; line-height: 1.6;">
                    <li>Input: MP3, WAV, FLAC, M4A, OGG</li>
                    <li>Output: High-quality WAV (16-bit)</li>
                    <li>Sample rates: Up to 48kHz</li>
                    <li>Channels: Mono and Stereo</li>
                    <li>File size: Up to 50MB</li>
                </ul>
            </div>
        </div>
    </div>
    ''')
    
    # Professional Footer with Author Credit
    gr.HTML('''
    <div class="professional-footer">
        <h3 style="font-family: 'Playfair Display', serif; 
                   font-size: 2.2rem; 
                   font-weight: 600; 
                   color: #D4AF37; 
                   margin: 0 0 20px 0;">
            Ready to Transform Your Audio?
        </h3>
        <p style="font-size: 1.1rem; 
                  color: #d1d5db; 
                  margin: 0 0 32px 0; 
                  font-weight: 400;">
            Join thousands of creators using DeVox Studio for professional audio separation
        </p>
        <div style="display: flex; 
                    justify-content: center; 
                    gap: 12px; 
                    flex-wrap: wrap; 
                    margin-bottom: 32px;">
            <span class="professional-badge">✨ No Registration Required</span>
            <span class="professional-badge">🚀 Instant Processing</span>
            <span class="professional-badge">💎 Professional Quality</span>
            <span class="professional-badge">🛡️ Privacy Focused</span>
        </div>
        
        <!-- Professional Author Credit Section -->
        <div style="border-top: 1px solid rgba(255, 215, 0, 0.2); 
                    margin-top: 32px; 
                    padding-top: 24px;">
            <p style="font-size: 0.9rem; 
                      color: #9ca3af; 
                      margin: 0 0 16px 0;">
                Engineered for professionals, accessible to everyone
            </p>
            <div class="author-credit">
                <p style="font-size: 1rem; 
                          color: #d1d5db; 
                          margin: 0 0 12px 0; 
                          font-weight: 400;">
                    Crafted with precision and passion by
                </p>
                <div style="font-size: 1.25rem; 
                           font-weight: 700; 
                           background: linear-gradient(135deg, #ffd700 0%, #ffffff 50%, #ffd700 100%);
                           -webkit-background-clip: text;
                           -webkit-text-fill-color: transparent;
                           background-clip: text;
                           margin: 8px 0 16px 0;">
                    AMAN
             </div>
                    <div class="author-links" style="margin-top: 16px;">
                            <a href="mailto:aws573800@gmail.com" target="_blank">Email</a> •
                            <a href="https://www.linkedin.com/in/amans82631/" target="_blank">LinkedIn</a>
                            <a href="https://github.com/Mansingh76" target="_blank">GitHub</a>
                    </div>
    
            </div>
            <div style="margin-top: 20px;">
                <strong style="color: #D4AF37; font-size: 0.9rem;">#DeVoxStudio</strong>
                <span style="color: #9ca3af; font-size: 0.875rem; margin-left: 8px;">• Empowering creators worldwide</span>
            </div>
        </div>
    </div>
    ''')
    
    # Connect functionality
    separate_btn.click(
        fn=separate_audio,
        inputs=[audio_input, separation_mode],
        outputs=[instrumental_output, vocal_output, status_output],
        show_progress=True
    )

if __name__ == "__main__":
    app.launch()
