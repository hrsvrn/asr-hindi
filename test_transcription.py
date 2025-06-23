#!/usr/bin/env python3
"""
Test script for debugging transcription issues
"""

import sys
import time
from inference_wrapped import transcribe_from_mic

def test_transcription():
    """Test the transcription functionality with debugging"""
    
    print("🎙️ Hindi Speech Transcription Test")
    print("=" * 50)
    
    try:
        print("📋 Instructions:")
        print("1. Speak clearly in Hindi")
        print("2. Stay close to the microphone")
        print("3. Avoid background noise")
        print()
        
        # Test with different durations
        durations = [3, 5]
        
        for duration in durations:
            print(f"\n🧪 Test {durations.index(duration) + 1}: {duration} second recording")
            print("-" * 30)
            
            print("⏰ Starting recording in 2 seconds...")
            time.sleep(2)
            
            result = transcribe_from_mic(duration=duration)
            
            print(f"\n📝 Result: {result}")
            print(f"📏 Length: {len(result) if result else 0} characters")
            
            if result.startswith("[ERROR]"):
                print("❌ Error detected!")
                break
            elif result.startswith("[WARNING]"):
                print("⚠️ Warning detected!")
            elif result.startswith("[INFO]"):
                print("ℹ️ Info message")
            else:
                print("✅ Transcription successful!")
            
            if durations.index(duration) < len(durations) - 1:
                print("\n⏳ Next test in 3 seconds...")
                time.sleep(3)
        
        print("\n🏁 Test completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_transcription() 