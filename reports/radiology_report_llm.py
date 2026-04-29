import os
import torch
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available.")

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    print("Warning: deep-translator not available.")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not available.")

from config import MED_GEMMA_MODEL_PATH, HUGGINGFACE_TOKEN, GEMINI_API_KEY, REPORT_GENERATION_MODE


class RadiologyReportGenerator:
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.translator = 'google' if TRANSLATOR_AVAILABLE else None  # deep-translator uses service name
        self.model = None
        self.tokenizer = None
        self.gemini_model = None
        
        if GEMINI_AVAILABLE and GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_models = [
                'gemini-2.0-flash-exp',
                'gemini-1.5-flash-latest',
                'gemini-1.5-flash-002',
                'gemini-1.5-flash',
                'gemini-1.5-pro-latest',
                'gemini-pro',
            ]
            
            for model_name in gemini_models:
                try:
                    self.gemini_model = genai.GenerativeModel(model_name)
                    print(f"✅ Gemini API configured: {model_name}")
                    break
                except Exception as e:
                    print(f"⚠️ Model {model_name} not available: {e}")
                    continue
        
        if TRANSFORMERS_AVAILABLE:
            self._load_medgemma_model()
    
    def _load_medgemma_model(self):
        try:
            print(f"Loading Med-Gemma from {MED_GEMMA_MODEL_PATH}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                MED_GEMMA_MODEL_PATH,
                trust_remote_code=True,
                token=HUGGINGFACE_TOKEN
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                MED_GEMMA_MODEL_PATH,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                token=HUGGINGFACE_TOKEN
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print("✅ Med-Gemma loaded successfully!")
            
        except Exception as e:
            print(f"⚠️ Error loading Med-Gemma: {e}")
            self.model = None
    
    def _create_comprehensive_prompt(self, kl_grade, confidence, patient_age=None, 
                                     patient_sex=None, clinical_history=None):
        
        kl_reference = {
            0: "No osteoarthritis - normal joint",
            1: "Doubtful osteoarthritis - minimal changes",
            2: "Mild osteoarthritis - definite changes",
            3: "Moderate osteoarthritis - significant changes",
            4: "Severe osteoarthritis - advanced degeneration"
        }
        
        patient_profile = []
        if patient_age:
            patient_profile.append(f"{patient_age}-year-old")
        if patient_sex:
            patient_profile.append(patient_sex)
        patient_profile.append("patient")
        if clinical_history:
            patient_profile.append(f"with {clinical_history}")
        patient_text = " ".join(patient_profile)
        
        prompt = f"""You are an expert radiologist writing a comprehensive knee X-ray report. Generate a complete, professional radiology report with ALL sections below.

PATIENT INFORMATION:
{patient_text}

AI CLASSIFICATION RESULT:
- Kellgren-Lawrence Grade: {kl_grade}
- AI Confidence: {confidence:.1%}
- Classification: {kl_reference.get(kl_grade, "Unknown")}

Generate a COMPLETE radiology report with these sections. Write detailed medical content for each section:

1. DISCLAIMER:
Write a standard AI-generated report disclaimer (2-3 sentences).

2. CLINICAL HISTORY:
Describe patient presentation and indication for examination (2-3 sentences).

3. TECHNIQUE:
Describe radiographic technique, views obtained, and AI analysis method (2-3 sentences).
Include: "AI Classification Confidence: {confidence:.1%}" and "Kellgren-Lawrence Grading System Applied"

4. FINDINGS:
Write detailed radiographic findings based on KL Grade {kl_grade} (4-6 sentences):
- Joint space assessment
- Osteophyte presence and distribution
- Subchondral bone changes
- Joint alignment and deformity
- Soft tissue observations

5. IMPRESSION:
State the radiographic diagnosis and severity (2-3 sentences).

6. RECOMMENDATIONS:
Provide clinical management recommendations appropriate for Grade {kl_grade} (2-3 sentences).

7. RISK PROFILE:
Describe the risk of disease progression, factors affecting prognosis, and long-term outlook for Grade {kl_grade} (3-4 sentences).

8. ESSENTIAL PRECAUTIONS:
List specific precautions and activities to avoid for Grade {kl_grade}. Include daily activity modifications, warning signs to monitor, and when to seek care (4-5 key points).

9. TREATMENT OPTIONS:
Describe available treatment modalities appropriate for Grade {kl_grade}. Include conservative management, medications, injections, physical therapy, and surgical options if applicable (5-6 treatment approaches).

10. FOLLOW-UP AND MONITORING:
Specify follow-up schedule, imaging frequency, monitoring parameters, and expected outcomes for Grade {kl_grade} (3-4 sentences).

Write each section with detailed, specific medical content. Use professional medical language. Make recommendations evidence-based and tailored to KL Grade {kl_grade}.

Generate the complete report now:"""
        
        return prompt
    
    def _generate_with_gemini(self, kl_grade, confidence, patient_age=None,
                             patient_sex=None, clinical_history=None):
        if not self.gemini_model:
            return None
        
        try:
            print("🤖 Generating report with Gemini API...")
            prompt = self._create_comprehensive_prompt(
                kl_grade, confidence, patient_age, patient_sex, clinical_history
            )
            
            generation_config = {
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40,
                'max_output_tokens': 2048,
            }
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            report = response.text.strip()
            
            if len(report) > 500:
                print(f"✅ Gemini generated {len(report)} chars")
                return report
            else:
                print("⚠️ Gemini report too short")
                return None
                
        except Exception as e:
            print(f"⚠️ Gemini generation error: {e}")
            return None
    
    def _generate_with_medgemma(self, kl_grade, confidence, patient_age=None,
                                patient_sex=None, clinical_history=None):
        if not self.model:
            return None
        
        try:
            print("🤖 Generating report with Med-Gemma...")
            prompt = self._create_comprehensive_prompt(
                kl_grade, confidence, patient_age, patient_sex, clinical_history
            )
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=800,
                    min_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.92,
                    top_k=40,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if prompt in generated_text:
                report = generated_text.replace(prompt, "").strip()
            else:
                report = generated_text[len(prompt):].strip()
            
            if len(report) > 200:
                print(f"✅ Med-Gemma generated {len(report)} chars")
                return report
            else:
                print("⚠️ Med-Gemma report too short")
                return None
                
        except Exception as e:
            print(f"⚠️ Med-Gemma generation error: {e}")
            return None
    
    def _generate_minimal_fallback(self, kl_grade, confidence, patient_age=None,
                                   patient_sex=None, clinical_history=None):
        print("⚠️ ALL LLMs FAILED - Using minimal emergency fallback")
        
        patient_text = f"{patient_age}-year-old {patient_sex}" if patient_age and patient_sex else "Patient"
        
        kl_names = {
            0: "No Osteoarthritis",
            1: "Doubtful Osteoarthritis", 
            2: "Mild Osteoarthritis",
            3: "Moderate Osteoarthritis",
            4: "Severe Osteoarthritis"
        }
        
        return f"""DISCLAIMER:
This is an AI-generated preliminary report. Requires confirmation by board-certified radiologist.

CLINICAL HISTORY:
{patient_text} presenting for knee X-ray evaluation.

TECHNIQUE:
Standard knee radiographs obtained. AI classification performed.
AI Classification Confidence: {confidence:.1%}
Kellgren-Lawrence Grading System Applied

FINDINGS:
Radiographic examination demonstrates findings consistent with Kellgren-Lawrence Grade {kl_grade}.

IMPRESSION:
{kl_names.get(kl_grade, "Osteoarthritis")} (Kellgren-Lawrence Grade {kl_grade}).

RECOMMENDATIONS:
Clinical correlation and orthopedic consultation recommended.

RISK PROFILE:
Risk assessment requires clinical evaluation.

ESSENTIAL PRECAUTIONS:
Follow standard osteoarthritis precautions. Consult healthcare provider.

TREATMENT OPTIONS:
Treatment plan should be determined by orthopedic specialist based on clinical evaluation.

FOLLOW-UP AND MONITORING:
Follow-up as recommended by treating physician.

---
Generated by AI System - Requires Professional Review"""
    
    def translate_report(self, report_text, target_language):
        """
        Translate report to target language using GoogleTranslator or Gemini as fallback
        """
        if target_language == 'en' or target_language is None:
            return report_text
        
        language_names = {
            'es': 'Spanish', 'fr': 'French', 'de': 'German', 'hi': 'Hindi',
            'ta': 'Tamil', 'te': 'Telugu', 'kn': 'Kannada', 'ml': 'Malayalam',
            'zh': 'Chinese', 'ja': 'Japanese', 'ar': 'Arabic', 'pt': 'Portuguese',
            'ru': 'Russian', 'ko': 'Korean', 'it': 'Italian', 'nl': 'Dutch',
            'bn': 'Bengali', 'mr': 'Marathi', 'gu': 'Gujarati', 'pa': 'Punjabi'
        }
        language_name = language_names.get(target_language, target_language)
        
        if self.translator:
            try:
                print(f"🔄 Translating to {language_name} via GoogleTranslator...")
                translator = GoogleTranslator(source='auto', target=target_language)
                translated = translator.translate(report_text)
                print("✅ Translation completed")
                return translated
            except Exception as e:
                print(f"⚠️ GoogleTranslator failed: {e}")
        
        if self.gemini_model:
            try:
                print(f"🔄 Translating to {language_name} via Gemini...")
                prompt = f"""Translate this medical radiology report to {language_name}.
Maintain all medical terminology accuracy and formatting.

Report:
{report_text}

Provide ONLY the translated text:"""
                
                response = self.gemini_model.generate_content(prompt)
                print("✅ Translation completed")
                return response.text.strip()
            except Exception as e:
                print(f"⚠️ Gemini translation failed: {e}")
        
        print("⚠️ All translation failed, returning English")
        return report_text
    
    def generate_report(self, kl_grade, confidence, patient_age=None,
                       patient_sex=None, clinical_history=None):
        """
        Main report generation - tries LLMs in order, minimal fallback only if all fail
        
        Priority:
        1. Gemini API (best for comprehensive generation)
        2. Med-Gemma (local model)
        3. Minimal emergency fallback
        """
        # Try Gemini first
        report = self._generate_with_gemini(
            kl_grade, confidence, patient_age, patient_sex, clinical_history
        )
        if report:
            return report
        
        # Try Med-Gemma
        report = self._generate_with_medgemma(
            kl_grade, confidence, patient_age, patient_sex, clinical_history
        )
        if report:
            return report
        
        # Emergency fallback
        print("⚠️ All LLM generation failed! Using minimal fallback")
        return self._generate_minimal_fallback(
            kl_grade, confidence, patient_age, patient_sex, clinical_history
        )
    
    def generate_complete_report(self, kl_grade, confidence, language='en',
                                patient_age=None, patient_sex=None, clinical_history=None):
        print(f"📝 Generating radiology report for KL Grade {kl_grade}...")
        
        english_report = self.generate_report(
            kl_grade, confidence, patient_age, patient_sex, clinical_history
        )
        
        if language != 'en':
            translated_report = self.translate_report(english_report, language)
        else:
            translated_report = english_report
        
        return english_report, translated_report


def generate_radiology_report(kl_grade, confidence, language='en',
                              patient_age=None, patient_sex=None, clinical_history=None):
    generator = RadiologyReportGenerator()
    return generator.generate_complete_report(
        kl_grade, confidence, language, patient_age, patient_sex, clinical_history
    )


if __name__ == '__main__':
    print("Testing LLM-First Radiology Report Generator")
    print("=" * 60)
    
    for grade in [0, 2, 4]:
        print(f"\n Testing KL Grade {grade}:")
        english, translated = generate_radiology_report(
            kl_grade=grade,
            confidence=0.87,
            language='en',
            patient_age=65,
            patient_sex='Male',
            clinical_history='knee pain for 6 months'
        )
        print(english[:300] + "...")
