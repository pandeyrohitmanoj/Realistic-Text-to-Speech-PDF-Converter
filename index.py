#!/usr/bin/env python3
"""
Multi-Speaker Text-to-Speech application with emotion detection
Uses functional programming approach with GPU acceleration
"""

import os
import gc
from pdf_cleaning import extract_clean_text_from_pdf
import shutil
import torch
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from TTS.api import TTS as CoquiTTS
import PyPDF2
import logging
import re
from transformers.pipelines import pipeline
import os
torch.cuda.set_per_process_memory_fraction(0.8) 

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize() 
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
os.environ['COQUI_TOS_AGREED'] = '1'
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from merge import combine_audio_with_metadata 
def verify_gpu_support() -> str:
    """Verify and return the best available device"""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
    else:
        device = "cpu"
        logger.warning("CUDA not available, using CPU")
    return device

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            index =0
            for page in pdf_reader.pages:
                index +=1
                if index < 50 : pass
                if index>100: break
                text += page.extract_text() + "\n"
        
        logger.info(f"Extracted {len(text)} characters from {pdf_path}")
        return text.strip()
    
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
        return ""

def clean_text_for_tts(text: str) -> str:
    """Clean and prepare text for TTS processing"""
    # Remove extra whitespace and newlines
    text = ' '.join(text.split())
    
    # Remove special characters that might cause issues
    text = text.replace('\n', ' ').replace('\t', ' ')
    
    # Fix common PDF extraction issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between lowercase-uppercase
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
    
    return text

def split_text_into_chunks(text: str, max_length: int = 125) -> List[str]:
    """Split text into proper sentence-based chunks"""
    # Split by sentence endings but keep the punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # If adding this sentence would exceed max_length
        # if len(current_chunk) + len(sentence) + 1 > max_length:
        #     if current_chunk:  # Save current chunk if it exists
        #         chunks.append(current_chunk.strip())
        #         current_chunk = sentence
        #     else:  # Single sentence is too long, split by commas
        if len(sentence) > max_length:
            parts = sentence.split(', ')
            temp_chunk = ""
            for part in parts:
                if len(temp_chunk) + len(part) + 2 < max_length:
                    temp_chunk += part + ", "
                else:
                    if temp_chunk:
                        chunks.append(temp_chunk.strip().rstrip(','))
                        temp_chunk=''
                    temp_chunk = part + ", "
            if temp_chunk:
                current_chunk = temp_chunk.strip().rstrip(',')
        else:
            current_chunk = sentence.strip()
        chunks.append(current_chunk)
    
    
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks

def get_speaker_profiles() -> Dict:
    """Define diverse speaker profiles with different characteristics"""
    return {
        "narrator": {
            "gender": "male",
            "age": "middle_aged",
            "ethnicity": "american",
            "voice_file": "voices/narrator_male_40s.wav",
            "description": "Professional narrator, clear and authoritative",
            "emotion_range": ["neutral", "professional", "serious"]
        },
        "young_female": {
            "gender": "female", 
            "age": "young",
            "ethnicity": "american",
            "voice_file": "voices/female_20s_energetic.wav",
            "description": "Young, enthusiastic female voice",
            "emotion_range": ["happy", "excited", "friendly"]
        },
        "old_wise_male": {
            "gender": "male",
            "age": "elderly", 
            "ethnicity": "british",
            "voice_file": "voices/male_60s_wise.wav",
            "description": "Elderly, wise, slow-paced speaker",
            "emotion_range": ["calm", "thoughtful", "serious"]
        },
        "professional_female": {
            "gender": "female",
            "age": "middle_aged",
            "ethnicity": "american", 
            "voice_file": "voices/female_35_business.wav",
            "description": "Professional businesswoman voice",
            "emotion_range": ["professional", "confident", "serious"]
        },
        "child_voice": {
            "gender": "female",
            "age": "child",
            "ethnicity": "american",
            "voice_file": "voices/child_girl_8.wav", 
            "description": "Child voice for young characters",
            "emotion_range": ["happy", "excited", "curious"]
        },
        "dramatic_male": {
            "gender": "male",
            "age": "middle_aged",
            "ethnicity": "theatrical",
            "voice_file": "voices/male_actor_dramatic.wav",
            "description": "Dramatic, expressive male voice",
            "emotion_range": ["angry", "sad", "excited", "fear"]
        },
        "gentle_female": {
            "gender": "female",
            "age": "young_adult", 
            "ethnicity": "soft_spoken",
            "voice_file": "voices/female_gentle_caring.wav",
            "description": "Gentle, caring female voice",
            "emotion_range": ["calm", "loving", "concerned"]
        },
        "child_boy.wav":{
            "gender": "male",
            "age": "child", 
            "ethnicity": "soft_spoken",
            "voice_file": "voices/female_gentle_caring.wav",
            "description": "Gentle, caring male voice",
            "emotion_range": ["calm", "loving", "confident"]
        },
    }

def initialize_sentiment_analyzer(device: str):
    """Initialize emotion detection model"""
    try:
        logger.info("Loading emotion detection model...")
        sentiment_analyzer = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=0 if device == "cuda" else -1,
            return_all_scores=True
        )
        logger.info("Emotion analyzer loaded successfully")
        return sentiment_analyzer
    
    except Exception as e:
        logger.error(f"Error loading emotion model: {e}")
        return 

def analyze_text_emotion(text: str, sentiment_analyzer) -> Tuple[str, float]:
    """Analyze emotion from text"""
    if sentiment_analyzer is None:
        return "neutral", 0.5
    
    try:
        clean_text = text.strip()[:512]
        if not clean_text:
            return "neutral", 0.5
        
        results = sentiment_analyzer(clean_text)
        
        # Handle emotion model output
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], list):  # Multiple emotions (emotion model)
                top_emotion = max(results[0], key=lambda x: x['score'])
                emotion_map = {
                    'joy': 'happy',
                    'sadness': 'sad', 
                    'anger': 'angry',
                    'fear': 'concerned',
                    'surprise': 'excited',
                    'disgust': 'serious',
                    'neutral': 'calm'
                }
                emotion = emotion_map.get(top_emotion['label'].lower(), 'neutral')
                confidence = top_emotion['score']
            else:  # Simple sentiment (sentiment model)
                emotion = 'happy' if results[0]['label'] == 'POSITIVE' else 'serious'
                confidence = results[0]['score']
        else:
            emotion, confidence = "neutral", 0.5
            
        return emotion, confidence
    
    except Exception as e:
        logger.error(f"Error in emotion analysis: {e}")
        return "neutral", 0.5

def detect_speaker_from_context(text: str, speakers: Dict) -> str:
    """Intelligently detect which speaker should read the text"""
    text_lower = text.lower()
    speaker='narrator'
    # Dialogue detection patterns
    if '"' in text or "'" in text:
        # Check for character indicators
        if any(word in text_lower for word in ['said', 'whispered', 'shouted', 'asked']):
            if any(word in text_lower for word in ['child', 'kid', 'little']):
                return "child_voice"
            elif any(word in text_lower for word in ['woman', 'she', 'her', 'lady']):
                return "young_female" if any(word in text_lower for word in ['young', 'girl']) else "professional_female"
            else:
                return "dramatic_male"
    
    # Content-based speaker selection
    if any(word in text_lower for word in ['analysis', 'report', 'study', 'data']):
        speaker= "professional_female"
    
    if any(word in text_lower for word in ['story', 'once upon', 'tale']):
        speaker= "gentle_female"
    
    if any(word in text_lower for word in ['wisdom', 'experience', 'years ago']):
        speaker= "old_wise_male"
    
    if any(word in text_lower for word in ['exciting', 'amazing', 'wow', 'cool']):
        speaker= "young_female"
    # Default narrator
    return speaker

def get_emotional_voice_settings(emotion: str, confidence: float) -> Dict:
    """Get voice parameters based on detected emotion"""
    emotion_settings = {
        "happy": {"speed": 1.1, "pitch_modifier": 0.1, "energy": 1.2},
        "excited": {"speed": 1.3, "pitch_modifier": 0.2, "energy": 1.4}, 
        "sad": {"speed": 0.8, "pitch_modifier": -0.1, "energy": 0.7},
        "angry": {"speed": 1.2, "pitch_modifier": 0.05, "energy": 1.3},
        "concerned": {"speed": 0.9, "pitch_modifier": -0.05, "energy": 0.9},
        "serious": {"speed": 0.95, "pitch_modifier": -0.02, "energy": 1.0},
        "calm": {"speed": 1.0, "pitch_modifier": 0.0, "energy": 1.0},
        "professional": {"speed": 0.95, "pitch_modifier": -0.02, "energy": 1.0}
    }
    
    base_settings = emotion_settings.get(emotion, emotion_settings["calm"])
    
    # Adjust intensity based on confidence
    intensity = min(confidence * 1.5, 1.0)  # Scale confidence
    
    return {
        "speed": 1.0 + (base_settings["speed"] - 1.0) * intensity,
        "pitch_modifier": base_settings["pitch_modifier"] * intensity,
        "energy": 1.0 + (base_settings["energy"] - 1.0) * intensity
    }

def initialize_tts_model(device: str) -> CoquiTTS | None:
    """Initialize TTS model with GPU support and proper error handling"""
    
    # Set environment variables to handle the safetensors issue
    os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/.cache/huggingface/transformers')
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    
    try:
        # Primary model with trust_remote_code and proper device handling
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        
        logger.info(f"Loading TTS model: {model_name}")
        logger.info(f"Device: {device}")
        
        # Initialize with explicit parameters
        tts = CoquiTTS(
            model_name=model_name,
            gpu=(device == "cuda"),
            # Add these parameters to handle the weights issue
        )
        
        # Test the model with a simple generation
        logger.info("Testing TTS model...")
        logger.info("TTS model loaded and tested successfully")
        return tts
    
    except Exception as e:
        logger.error(f"Error loading primary TTS model: {e}")
        logger.info("Trying alternative initialization method...")
        
        # Alternative method - manual model loading
        try:
            # Force download and trust the checkpoint
            import huggingface_hub
            
            # Clear any cached problematic files
            cache_dir = os.path.expanduser('~/.local/share/tts')
            if os.path.exists(cache_dir):
                logger.info("Clearing TTS cache...")
                import shutil
                shutil.rmtree(cache_dir, ignore_errors=True)
            
            # Reinitialize with clean cache
            tts = CoquiTTS(model_name, gpu=(device == "cuda"))
            logger.info("TTS model loaded with alternative method")
            return tts
            
        except Exception as e2:
            logger.error(f"Alternative method failed: {e2}")
            logger.info("Trying fallback models...")
            
            # Try multiple fallback models
            fallback_models = [
                "tts_models/multilingual/multi-dataset/xtts_v1",  # Older XTTS
                "tts_models/en/ljspeech/tacotron2-DDC",          # Simple English
                "tts_models/en/ljspeech/glow-tts",               # Another fallback
                "tts_models/en/ljspeech/speedy-speech",          # Fast model
            ]
            
            for fallback_model in fallback_models:
                try:
                    logger.info(f"Trying fallback model: {fallback_model}")
                    tts = CoquiTTS(fallback_model, gpu=(device == "cuda"))
                    logger.info(f"Successfully loaded fallback model: {fallback_model}")
                    return tts
                except Exception as e3:
                    logger.warning(f"Fallback model {fallback_model} failed: {e3}")
                    continue
            
            logger.error("All TTS models failed to load")
            return None

def text_to_speech_with_emotion(tts_model: CoquiTTS, text: str, output_path: str, 
                               speaker_profile: Dict, emotion: str, confidence: float,
                               device: str) -> bool:
    """Convert text to speech with emotional and speaker control"""
    if tts_model is None:
        logger.error("TTS model not initialized")
        return False
    
    try:
        logger.info(f"Converting with {speaker_profile['description']}, emotion: {emotion} ({confidence:.2f})")
        
        # Get emotional voice settings
        voice_settings = get_emotional_voice_settings(emotion, confidence)
        
        # Check if speaker voice file exists
        speaker_wav = speaker_profile.get("voice_file")
        if speaker_wav and os.path.exists(speaker_wav):
            wav = tts_model.tts(
                text=text,
                speaker_wav=speaker_wav,
                language="en"
            )
        else:
            # Fallback to default voice
            logger.warning(f"Speaker file not found: {speaker_wav}, using default")
            wav = tts_model.tts(text=text)
        
        # Apply emotional modifications (speed adjustment)
        if voice_settings["speed"] != 1.0:
            try:
                import librosa
                import numpy as np
                
                # Convert to numpy array first if needed
                if isinstance(wav, list):
                    wav = np.array(wav, dtype=np.float32)
                elif hasattr(wav, 'cpu'):  # PyTorch tensor
                    wav = wav.cpu().numpy().astype(np.float32)
                
                # Apply time stretch
                wav = librosa.effects.time_stretch(wav, rate=1/voice_settings["speed"])
                
            except ImportError:
                logger.warning("librosa not available for speed modification")
            except Exception as e:
                logger.warning(f"Speed modification failed: {e}")
        # Save audio file
        sf.write(output_path, wav, 22050)
        
        logger.info(f"Audio saved: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error in TTS conversion: {e}")
        return False

def process_pdf_with_multiple_speakers(text: str, output_dir: str = "output") -> List[Dict]:
    """Process PDF with intelligent speaker and emotion detection"""
    # Setup
    device = verify_gpu_support()
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Initialize models
    tts_model = initialize_tts_model(device)
    sentiment_analyzer = initialize_sentiment_analyzer(device)
    speakers = get_speaker_profiles()
    
    clean_text = clean_text_for_tts(text)
    text_chunks = split_text_into_chunks(clean_text, 125)
    
    # Process each chunk with speaker and emotion detection
    output_files = []
    
    for i, chunk in enumerate(text_chunks):
        # Detect appropriate speaker
        speaker_key = detect_speaker_from_context(chunk, speakers)
        speaker_profile = speakers[speaker_key]
        
        # Analyze emotion
        emotion, confidence = analyze_text_emotion(chunk, sentiment_analyzer)
        
        # Generate filename with metadata
        output_path = f"{output_dir}/{i+1:03d}.wav"
        if tts_model == None:
            return []
        # Convert to speech
        success = text_to_speech_with_emotion(
            tts_model=tts_model,
            text=chunk,
            output_path=output_path,
            speaker_profile=speaker_profile,
            emotion=emotion,
            confidence=confidence,
            device=device
        )
        
        if success:
            output_files.append({
                "file": output_path,
                "speaker": speaker_key,
                "emotion": emotion,
                "confidence": confidence,
                "text_preview": chunk[:50] + "..." if len(chunk) > 50 else chunk
            })
        
        logger.info(f"Processed {i+1}/{len(text_chunks)}: {speaker_key} ({emotion})")
    del tts_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize() 
    
    return output_files

def create_speaker_voice_samples():
    """Generate sample voice files for each speaker (placeholder function)"""
    voices_dir = Path("voices")
    voices_dir.mkdir(exist_ok=True)
    
    sample_files = [
        "narrator_male_40s.wav",
        "female_20s_energetic.wav", 
        "male_60s_wise.wav",
        "female_35_business.wav",
        "child_girl_8.wav",
        "male_actor_dramatic.wav",
        "female_gentle_caring.wav",
        "child_boy.wav"
    ]
    
    print("To use custom speakers, place these voice sample files in 'voices/' directory:")
    for file in sample_files:
        print(f"  - {file} (10-30 seconds of clean speech)")
    
    print("\nOr download from: https://commonvoice.mozilla.org/")
    print("Or record your own samples with different speakers")
data_dir = Path.cwd() / "data"
def get_pdf_files_pathlib(directory_path=str(data_dir)):
    directory = Path(directory_path)
    
    return [
        {
            'filename': pdf_file.stem,
            'path': str(pdf_file)
        }
        for pdf_file in directory.rglob('*.pdf')
    ]



def main():
    """Enhanced main function with multi-speaker support"""
    input_dir = "output" 
    shutil.rmtree(input_dir,ignore_errors=True)
    files = get_pdf_files_pathlib()
    pdf_file = files[0]['path']
    text = extract_clean_text_from_pdf(pdf_file)
    print(text)
    if not Path(pdf_file).exists():
        print("PDF file 'test.pdf' not found!")
        print("Please place a PDF file named 'test.pdf' in the current directory")
        return
    
    print(" Multi-Speaker TTS with Emotion Detection")
    print("=" * 50)
    
    if not Path("voices").exists():
        print("Setting up voice samples directory...")
        create_speaker_voice_samples()
        print("\nUsing default voices for now. Add custom samples later for better results.")
    
    results = process_pdf_with_multiple_speakers(text)
    if results:
        print(f"\n Generated {len(results)} audio files:")
        print("-" * 60)
        
        speaker_count = {}
        emotion_count = {}
        
        for result in results:
            speaker = result['speaker']
            emotion = result['emotion']
            
            speaker_count[speaker] = speaker_count.get(speaker, 0) + 1
            emotion_count[emotion] = emotion_count.get(emotion, 0) + 1
            
            print(f"{result['file']}")
            print(f"Speaker: {speaker} | 😊 Emotion: {emotion} ({result['confidence']:.2f})")
            print(f"Text: {result['text_preview']}")
            print()
        
        print(" Summary:")
        print(f"Speakers used: {dict(speaker_count)}")
        print(f"Emotions detected: {dict(emotion_count)}")
        result = combine_audio_with_metadata(input_dir, files[0]['filename']+'.wav')
    
    else:
        print(" Failed to generate audio files")

# if __name__ == "__main__":
#     main()
# input_dir = Path.cwd() / "output" 
# files = get_pdf_files_pathlib(input_dir)
# pdf_file = files[0]['path']
# text = extract_clean_text_from_pdf(pdf_file)
# voices = Path.cwd() / 'voices'
# combine_audio_with_metadata( str(input_dir), 'result.wav')
