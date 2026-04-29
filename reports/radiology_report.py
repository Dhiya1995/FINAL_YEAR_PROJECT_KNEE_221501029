
import os
import torch
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Using template-based generation.")

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    print("Warning: deep-translator not available. Will use Gemini API for translation.")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not available. Translation fallback limited.")

from config import MED_GEMMA_MODEL_PATH, HUGGINGFACE_TOKEN, GEMINI_API_KEY, REPORT_GENERATION_MODE

class RadiologyReportGenerator:
    """
    Generate structured radiology reports using Med-Gemma LLM
    
    Workflow:
    1. Prompt Creation: Convert KL grade to structured text prompt
    2. LLM Integration: Med-Gemma generates English report with findings and recommendations
    3. Translation: Google Translate API converts report to target language
    4. Output: Structured report ready for PDF formatting
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.translator = 'google' if TRANSLATOR_AVAILABLE else None  # deep-translator uses service name
        self.model = None
        self.tokenizer = None
        self.gemini_model = None
        
        # Initialize Gemini API for translation fallback
        # Try multiple model versions for compatibility
        if GEMINI_AVAILABLE and GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_models = [
                'gemini-2.0-flash-exp',       # ✅ WORKS with your API key!
                'gemini-1.5-flash-latest',    # Fallback options below
                'gemini-1.5-flash-002',       
                'gemini-1.5-flash',           
                'gemini-1.5-pro-latest',      
                'gemini-pro',                 
            ]
            
            self.gemini_model = None
            for model_name in gemini_models:
                try:
                    self.gemini_model = genai.GenerativeModel(model_name)
                    # Don't test during initialization - just create the model object
                    print(f"✅ Gemini API configured with model: {model_name}")
                    break
                except Exception as e:
                    print(f"⚠️ Model {model_name} not available: {e}")
                    continue
            
            if not self.gemini_model:
                print("⚠️ All Gemini models failed to initialize. Translation fallback limited.")
        
        if TRANSFORMERS_AVAILABLE:
            self._load_medgemma_model()
    
    def _load_medgemma_model(self):
        """Load Med-Gemma model for medical report generation"""
        try:
            print(f"Loading Med-Gemma/Gemma model from {MED_GEMMA_MODEL_PATH}...")
            print("This may take a few minutes on first run...")
            
            # Use Hugging Face token for authentication
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
            print("✅ Med-Gemma model loaded successfully!")
            
        except Exception as e:
            print(f"⚠️ Error loading Med-Gemma model: {e}")
            print("Falling back to template-based report generation")
            self.model = None
    
    def _create_structured_prompt(self, kl_grade, confidence, patient_age=None, 
                                   patient_sex=None, clinical_history=None):
        """
        Create structured prompt for Med-Gemma/Phi-2
        Optimized for phi-2 instruction following
        """
        
        # KL Grade descriptions based on Kellgren-Lawrence classification
        kl_descriptions = {
            0: {
                "name": "No osteoarthritis",
                "findings": "Normal joint space width. No osteophytes. No subchondral sclerosis. No joint deformity.",
                "impression": "No radiographic evidence of knee osteoarthritis.",
                "recommendations": "No treatment required. Maintain healthy lifestyle and regular physical activity."
            },
            1: {
                "name": "Doubtful osteoarthritis",
                "findings": "Possible minimal joint space narrowing. Possible small osteophytes. Minimal to no subchondral sclerosis.",
                "impression": "Doubtful radiographic evidence of knee osteoarthritis. Early degenerative changes may be present.",
                "recommendations": "Conservative management and clinical follow-up recommended. Consider physical therapy."
            },
            2: {
                "name": "Mild osteoarthritis",
                "findings": "Definite joint space narrowing. Small osteophytes present at joint margins. Mild subchondral sclerosis may be present.",
                "impression": "Mild knee osteoarthritis with definite osteophytes and joint space narrowing.",
                "recommendations": "Physical therapy, weight management, and NSAIDs as appropriate. Orthopedic follow-up as needed."
            },
            3: {
                "name": "Moderate osteoarthritis",
                "findings": "Moderate joint space narrowing (25-50% reduction). Multiple moderate-sized osteophytes. Subchondral sclerosis present. Possible mild bone deformity.",
                "impression": "Moderate knee osteoarthritis with significant joint space narrowing and osteophyte formation.",
                "recommendations": "Orthopedic consultation for advanced treatment options. Consider intra-articular injections or bracing."
            },
            4: {
                "name": "Severe osteoarthritis",
                "findings": "Severe joint space narrowing (>50% reduction or bone-on-bone contact). Large osteophytes throughout joint. Marked subchondral sclerosis. Definite bone deformity and possible subluxation.",
                "impression": "Severe knee osteoarthritis with marked degenerative changes and joint deformity.",
                "recommendations": "Urgent orthopedic referral recommended. Surgical intervention (total knee arthroplasty) should be considered."
        }
        }
        
        grade_info = kl_descriptions[kl_grade]
        
        # Build clinical history section
        clinical_info = []
        if patient_age:
            clinical_info.append(f"{patient_age}-year-old")
        if patient_sex:
            clinical_info.append(patient_sex)
        clinical_info.append("patient")
        
        if clinical_history:
            clinical_info.append(f"with {clinical_history}")
        else:
            clinical_info.append("presenting with knee pain for evaluation")
        
        clinical_history_text = " ".join(clinical_info) + "."
        
        # Create structured prompt - LLM generates CONTENT, we provide STRUCTURE
        prompt = f"""You are a radiologist. Write the medical content for each section below. Write ONLY the medical findings, do NOT repeat the section headers. Write in complete sentences as natural medical prose.

Patient: {clinical_history_text}
Diagnosis: Kellgren-Lawrence Grade {kl_grade} - {grade_info['name']}
Model Confidence: {confidence:.1%}

Expected findings for this grade: {grade_info['findings']}
Clinical impression: {grade_info['impression']}
Treatment approach: {grade_info['recommendations']}

Now write detailed medical content for these sections (write content only, not headers):

Section 1 - Clinical History (2-3 sentences about patient presentation and indication):
Write about the patient's clinical presentation and reason for examination.

Section 2 - Technique (1-2 sentences about imaging method):
Describe the radiographic technique and views obtained.

Section 3 - Findings (4-5 sentences with detailed observations):
Describe joint space, osteophytes, subchondral changes, alignment, and soft tissues based on KL Grade {kl_grade}.

Section 4 - Impression (2-3 sentences with diagnosis):
State the radiographic diagnosis and severity.

Section 5 - Recommendations (2-3 sentences with clinical advice):
Provide appropriate management recommendations for this grade.

Write natural medical prose for each section:"""
        
        return prompt
    
    def _clean_code_artifacts(self, text):
        """
        Remove Python code artifacts and extract medical content
        Phi-2 tends to generate code - this cleans it up
        """
        import re
        
        # If text doesn't look like code, return as-is
        if not any(indicator in text for indicator in ['import ', 'def ', '```python', 'python', '# ', 'print(', 'f"""', 'f\'\'\'']):
            return text
        
        print("🧹 Detected code artifacts, cleaning...")
        
        # Remove code blocks wrapped in triple backticks
        text = re.sub(r'```python.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        
        # Try to extract content from f-strings or triple-quoted strings
        # Look for patterns like f"""CLINICAL HISTORY:...""" or """..."""
        string_patterns = [
            r'f?"""(.*?)"""',  # Triple quotes
            r"f?'''(.*?)'''",  # Triple single quotes
            r'f?"(.*?)"',      # Double quotes (multiline)
        ]
        
        extracted_texts = []
        for pattern in string_patterns:
            matches = re.findall(pattern, text, flags=re.DOTALL)
            for match in matches:
                # Only keep if it looks like medical content (has key medical terms)
                if len(match) > 100 and any(term in match.upper() for term in ['CLINICAL', 'HISTORY', 'TECHNIQUE', 'FINDINGS', 'IMPRESSION', 'RECOMMENDATION', 'PATIENT', 'KNEE', 'JOINT']):
                    extracted_texts.append(match.strip())
        
        # If we extracted medical text from strings, use it
        if extracted_texts:
            # Use the longest extracted text
            text = max(extracted_texts, key=len)
            print(f"✅ Extracted medical text from code ({len(text)} chars)")
        
        # Remove remaining code-like lines
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Skip obvious code lines
            if any(skip in line for skip in ['import ', 'from ', 'def ', '# Solution', '# Patient', 'patient_info = {', 'diagnosis = ', 'confidence = ']):
                continue
            # Skip lines that are just variable assignments
            if re.match(r'^\s*\w+\s*=\s*[{\[\'\"]', line):
                continue
            # Keep lines that look like medical text
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines).strip()
        
        # Final cleanup: remove multiple newlines, clean up whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'\\n', '\n', text)  # Fix literal \n
        text = text.strip()
        
        return text
    
    def _format_with_template_structure(self, llm_content, kl_grade, confidence, patient_age=None, 
                                       patient_sex=None, clinical_history=None):
        """
        Wrap LLM-generated content with professional template structure
        Structure from template, content from LLM
        """
        
        # Build patient info text
        demographics = []
        if patient_age:
            demographics.append(f"{patient_age}-year-old")
        if patient_sex:
            demographics.append(patient_sex)
        demographics.append("patient")
        patient_text = " ".join(demographics)
        
        # Parse LLM content (try to extract sections)
        sections = {
            'history': '',
            'technique': '',
            'findings': '',
            'impression': '',
            'recommendations': ''
        }
        
        # Try to intelligently parse the LLM output
        # Look for natural section breaks or use the whole text
        lines = llm_content.split('\n')
        current_section = 'history'
        section_order = ['history', 'technique', 'findings', 'impression', 'recommendations']
        section_index = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line indicates new section
            line_lower = line.lower()
            if any(word in line_lower for word in ['section 2', 'technique:', 'imaging method']):
                section_index = 1
                current_section = 'technique'
                continue
            elif any(word in line_lower for word in ['section 3', 'findings:', 'radiographic findings']):
                section_index = 2
                current_section = 'findings'
                continue
            elif any(word in line_lower for word in ['section 4', 'impression:', 'diagnosis:']):
                section_index = 3
                current_section = 'impression'
                continue
            elif any(word in line_lower for word in ['section 5', 'recommendation', 'management']):
                section_index = 4
                current_section = 'recommendations'
                continue
            
            # Add line to current section
            if line and not line.startswith('Section'):
                sections[current_section] += line + ' '
        
        # Clean up sections
        for key in sections:
            sections[key] = sections[key].strip()
        
        # If sections are empty, try to use the whole content intelligently
        if not any(sections.values()):
            # Use the whole content as findings
            sections['findings'] = llm_content.strip()
            sections['history'] = f"The {patient_text} presented for knee X-ray evaluation with complaints of knee discomfort."
            sections['technique'] = "Standard anteroposterior and lateral radiographic views of the knee were obtained."
            sections['impression'] = f"Findings consistent with Kellgren-Lawrence Grade {kl_grade} osteoarthritis."
            sections['recommendations'] = "Clinical correlation and appropriate orthopedic follow-up recommended."
        
        # Get grade-specific additional information
        kl_additional_info = self._get_grade_specific_info(kl_grade)
        
        # Build structured report with template headings + ADDITIONAL SECTIONS
        structured_report = f"""DISCLAIMER:
This report is generated using artificial intelligence analysis and should be considered preliminary. Definitive diagnosis requires in-person evaluation by a board-certified radiologist or orthopedic specialist.

CLINICAL HISTORY:
{sections['history'] if sections['history'] else f'The {patient_text} presented for knee X-ray evaluation with knee pain.'}

TECHNIQUE:
{sections['technique'] if sections['technique'] else 'Standard anteroposterior and lateral radiographic views of the knee were obtained and analyzed using AI-assisted classification system.'}
AI Classification Confidence: {confidence:.1%}
Kellgren-Lawrence Grading System Applied

FINDINGS:
{sections['findings'] if sections['findings'] else 'Radiographic examination demonstrates degenerative changes consistent with the classified grade.'}

IMPRESSION:
{sections['impression'] if sections['impression'] else f'Kellgren-Lawrence Grade {kl_grade} knee osteoarthritis.'}

RECOMMENDATIONS:
{sections['recommendations'] if sections['recommendations'] else 'Clinical correlation recommended. Orthopedic consultation as appropriate.'}

RISK PROFILE:
{kl_additional_info['risk_profile']}

ESSENTIAL PRECAUTIONS:
{kl_additional_info['precautions']}

TREATMENT OPTIONS:
{kl_additional_info['treatments']}

FOLLOW-UP AND MONITORING:
{kl_additional_info['followup']}

---
Generated by AI-Powered Knee Osteoarthritis Analysis System
Kellgren-Lawrence Grade: {kl_grade}
"""
        
        return structured_report.strip()
    
    def _get_grade_specific_info(self, kl_grade):
        """
        Get additional grade-specific information for comprehensive reporting
        Returns risk profile, precautions, treatments, and follow-up info
        """
        
        grade_info = {
            0: {
                'risk_profile': 'Low risk for developing osteoarthritis. Maintain healthy lifestyle and regular physical activity to preserve joint health. Genetic predisposition should be monitored if family history exists.',
                'precautions': 'Continue regular physical activity with proper form. Maintain ideal body weight. Use appropriate footwear during exercise. Consider preventive measures if family history of osteoarthritis exists. Annual check-ups recommended for individuals over 50 years.',
                'treatments': 'No specific treatment required at this time. Focus on prevention through regular low-impact exercise (swimming, cycling, walking). Maintain balanced diet rich in omega-3 fatty acids and antioxidants. Weight management to prevent future joint stress.',
                'followup': 'Routine health maintenance. Annual screening recommended if risk factors present. Contact healthcare provider if knee pain or swelling develops. Maintain active lifestyle and healthy weight.'
            },
            1: {
                'risk_profile': 'Moderate risk of progression to definite osteoarthritis. Risk factors include obesity, previous knee injury, family history, and high-impact activities. Early intervention may prevent progression.',
                'precautions': 'Avoid high-impact activities (running, jumping, heavy lifting). Use proper footwear with good cushioning. Apply ice after activities to reduce inflammation (15-20 minutes). Monitor symptoms regularly and report increased pain or stiffness. Consider knee-friendly exercises like swimming or cycling.',
                'treatments': 'Observation with regular follow-up imaging every 12-24 months. Physical therapy focusing on quadriceps strengthening. Low-impact aerobic exercises. Over-the-counter NSAIDs for occasional discomfort. Glucosamine and chondroitin supplements may be considered. Weight reduction if BMI > 25.',
                'followup': 'Follow-up imaging in 12-24 months or sooner if symptoms worsen. Monitor for progression of pain or limitation in daily activities. Physical therapy evaluation recommended. Weight management counseling if applicable.'
            },
            2: {
                'risk_profile': 'High risk of progression without intervention. Increased risk of developing moderate-to-severe osteoarthritis within 5-10 years. Activity modification and treatment adherence are critical to slow disease progression.',
                'precautions': 'Avoid prolonged standing or walking on hard surfaces. Use assistive devices (cane, walking stick) for long distances. Apply ice after activities (15-20 minutes). Avoid stairs when possible. Wear supportive, cushioned footwear. Consider knee braces for additional support during activity.',
                'treatments': 'Regular physical therapy 2-3 times per week. Prescription NSAIDs or COX-2 inhibitors as needed. Intra-articular corticosteroid injections (up to 3-4 per year). Hyaluronic acid viscosupplementation injections. Topical analgesics (diclofenac gel, capsaicin cream). Weight loss program if overweight (target 5-10% reduction). Low-impact exercises: swimming, water aerobics, stationary cycling.',
                'followup': 'Follow-up every 3-6 months to monitor progression. Repeat imaging annually or as clinically indicated. Physical therapy assessment and adjustment of exercise program. Pain management review. Consider specialist referral if conservative measures inadequate.'
            },
            3: {
                'risk_profile': 'Very high risk of progression to severe osteoarthritis requiring surgery. Without intervention, likely progression to Grade 4 within 3-5 years. Quality of life significantly impacted. Surgical consultation may be appropriate.',
                'precautions': 'Limit weight-bearing activities significantly. Use assistive devices regularly (walker if both knees affected). Apply heat before activity and ice after. Sleep with pillow between knees if lying on side. Consider occupational therapy for daily living adaptations. Use elevated toilet seat and shower chair to reduce joint stress.',
                'treatments': 'Intensive physical therapy with aquatic therapy component. Stronger prescription pain medications (tramadol, duloxetine). Regular intra-articular injections every 3-4 months. PRP (Platelet-Rich Plasma) injections may be considered. Unloader knee braces to shift weight from affected compartment. Surgical consultation for possible osteotomy, partial knee replacement, or total knee arthroplasty evaluation. Pain management specialist referral. Mandatory weight loss program if BMI > 30.',
                'followup': 'Follow-up every 2-3 months for close monitoring. Imaging every 6-12 months. Orthopedic consultation for surgical options discussion. Pain management optimization. Assessment of daily function and quality of life. Pre-surgical evaluation if conservative treatment fails.'
            },
            4: {
                'risk_profile': 'End-stage disease requiring surgical intervention for functional improvement. Non-surgical management provides only limited symptomatic relief. Significant disability without treatment. Quality of life severely impacted.',
                'precautions': 'Minimize all weight-bearing activities. Use assistive devices for all ambulation (walker, wheelchair for long distances). Fall prevention measures critical. Use adaptive equipment for daily activities. Avoid stairs entirely. Consider stair lift or bedroom relocation. Pain management critical - use prescribed medications as directed. Monitor for complications including depression and deconditioning.',
                'treatments': 'Urgent orthopedic referral for total knee arthroplasty (TKA) evaluation. Pre-surgical optimization: cardiac clearance, diabetes control, smoking cessation. Pain management: consider opioid alternatives, nerve blocks, radiofrequency ablation. Maximal medical therapy: NSAIDs, acetaminophen, topical agents. Final attempts at injection therapy if surgery not immediately possible. Comprehensive pre-hab physical therapy program. Nutritional counseling and weight optimization mandatory. Psychological support for chronic pain and surgical preparation.',
                'followup': 'Immediate orthopedic referral for surgical evaluation. Weekly to bi-weekly monitoring until surgery scheduled. Pain management optimization critical. Pre-operative preparation and education. Post-operative care planning. Disability evaluation and work accommodations if applicable. Social services referral for surgical preparation and home modifications.'
            }
        }
        
        return grade_info.get(kl_grade, grade_info[2])  # Default to Grade 2 if not found
    
    def generate_report_with_medgemma(self, kl_grade, confidence, patient_age=None,
                                      patient_sex=None, clinical_history=None):
        """
        Generate radiology report using Med-Gemma LLM or template based on configuration
        """
        
        # Check report generation mode from config
        if REPORT_GENERATION_MODE == 'template':
            print("📋 Using template-based report generation (configured mode)")
            return self._generate_template_report(kl_grade, confidence, patient_age, 
                                                  patient_sex, clinical_history)
        
        prompt = self._create_structured_prompt(
            kl_grade, confidence, patient_age, patient_sex, clinical_history
        )
        
        if self.model is None:
            if REPORT_GENERATION_MODE == 'llm':
                print("⚠️ LLM mode requested but Med-Gemma not available!")
            print("📋 Med-Gemma not available, using template...")
            return self._generate_template_report(kl_grade, confidence, patient_age, 
                                                  patient_sex, clinical_history)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate report with Med-Gemma
            print("Generating radiology report with Med-Gemma...")
            print(f"📝 Input prompt length: {len(prompt)} characters")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,          # Reduced for faster generation on CPU
                    min_new_tokens=50,           # Lower minimum for faster results
                    temperature=0.7,             # Balanced temperature for coherent output
                    do_sample=True,
                    top_p=0.92,                  # Nucleus sampling
                    top_k=40,                    # Reduced for more focused output
                    repetition_penalty=1.15,     # Prevent repetition
                    no_repeat_ngram_size=3,      # Prevent 3-gram repetition
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True          # Stop early if appropriate
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"🔍 Generated text length: {len(generated_text)} characters")
            
            # Extract report (remove prompt)
            # Try multiple extraction methods
            report = None
            
            # Method 1: Remove the exact prompt
            if prompt in generated_text:
                report = generated_text.replace(prompt, "").strip()
                print("📝 Extraction Method 1: Removed exact prompt")
            
            # Method 2: Split by the last instruction line
            elif "Generate a professional, clinical report following standard radiology reporting format:" in generated_text:
                parts = generated_text.split("Generate a professional, clinical report following standard radiology reporting format:")
                if len(parts) > 1:
                    report = parts[-1].strip()
                    print("📝 Extraction Method 2: Split by instruction")
            
            # Method 3: Just take everything after prompt length
            else:
                report = generated_text[len(prompt):].strip()
                print("📝 Extraction Method 3: Character offset")
            
            # Clean up the report
            if report:
                report = report.strip()
                print(f"📄 Extracted report length: {len(report)} characters")
                print(f"📄 Report preview: {report[:300]}...")
                
                # Post-process: Remove code artifacts
                report = self._clean_code_artifacts(report)
                print(f"📄 After cleaning: {len(report)} characters")
            
            # Only fallback to template if truly empty or extremely short
            if not report or len(report) < 50:
                print(f"⚠️ LLM generated empty or very short output ({len(report) if report else 0} chars)")
                if REPORT_GENERATION_MODE == 'llm':
                    print("❌ LLM mode enforced but generation failed - returning what we have")
                    if report:
                        return report
                    else:
                        return "ERROR: LLM failed to generate report. Please check model loading."
                else:
                    print("📋 Falling back to template-based generation...")
                    return self._generate_template_report(kl_grade, confidence, patient_age,
                                                         patient_sex, clinical_history)
            
            # HYBRID APPROACH: Wrap LLM content with template structure
            print("🏗️ Formatting LLM content with template structure...")
            structured_report = self._format_with_template_structure(
                report, kl_grade, confidence, patient_age, patient_sex, clinical_history
            )
            
            print(f"✅ Report generated successfully with Med-Gemma/Phi-2 ({len(structured_report)} characters)")
            print("   Structure: Template ✅ | Content: LLM-Generated ✅")
            return structured_report
            
        except Exception as e:
            print(f"⚠️ Error generating report with Med-Gemma: {e}")
            import traceback
            traceback.print_exc()
            
            if REPORT_GENERATION_MODE == 'llm':
                print("❌ LLM mode enforced but encountered error during generation")
                return f"ERROR: LLM report generation failed with error: {str(e)}"
            else:
                print("📋 Falling back to template-based generation...")
                return self._generate_template_report(kl_grade, confidence, patient_age,
                                                      patient_sex, clinical_history)
    
    def _generate_template_report(self, kl_grade, confidence, patient_age=None,
                                   patient_sex=None, clinical_history=None):
        """
        Fallback template-based report generation
        Ensures report is always generated even if Med-Gemma fails
        """
        
        # KL Grade specific findings
        findings_by_grade = {
            0: {
                "joint_space": "The joint space is preserved and symmetric.",
                "osteophytes": "No osteophytes are identified.",
                "sclerosis": "No subchondral sclerosis is present.",
                "other": "Joint alignment is maintained. No joint effusion or soft tissue abnormality.",
                "conclusion": [
                    "No radiographic evidence of knee osteoarthritis (Kellgren-Lawrence Grade 0).",
                    "Normal knee joint anatomy maintained.",
                    "Clinical correlation recommended if symptoms present."
                ]
            },
            1: {
                "joint_space": "The joint space appears minimally narrowed or possibly within normal limits.",
                "osteophytes": "Possible small osteophytes noted, though definitive identification is uncertain.",
                "sclerosis": "Minimal to no subchondral sclerosis.",
                "other": "Joint alignment is maintained. No significant soft tissue abnormality.",
                "conclusion": [
                    "Doubtful osteoarthritis (Kellgren-Lawrence Grade 1).",
                    "Early degenerative changes may be present but are not definitive.",
                    "Conservative management and clinical follow-up recommended.",
                    "Consider repeat imaging if symptoms progress."
                ]
            },
            2: {
                "joint_space": "Definite joint space narrowing is present, more pronounced in the medial compartment.",
                "osteophytes": "Small osteophytes are identified at the joint margins, particularly at the femoral and tibial margins.",
                "sclerosis": "Mild subchondral sclerosis is noted in the weight-bearing areas.",
                "other": "Overall joint alignment is maintained. No significant effusion.",
                "conclusion": [
                    "Mild knee osteoarthritis (Kellgren-Lawrence Grade 2).",
                    "Definite joint space narrowing and osteophyte formation present.",
                    "Recommend conservative management including physical therapy, weight management, and NSAIDs as appropriate.",
                    "Clinical correlation and orthopedic follow-up as needed."
                ]
            },
            3: {
                "joint_space": "Moderate joint space narrowing is evident, with 25-50% reduction compared to normal, predominantly affecting the medial compartment.",
                "osteophytes": "Multiple moderate-sized osteophytes are present at femoral and tibial margins.",
                "sclerosis": "Moderate subchondral sclerosis is present in weight-bearing regions.",
                "other": "Mild varus or valgus angulation may be developing. Possible small joint effusion.",
                "conclusion": [
                    "Moderate knee osteoarthritis (Kellgren-Lawrence Grade 3).",
                    "Significant joint space narrowing with multiple osteophytes and subchondral sclerosis.",
                    "Recommend orthopedic consultation for advanced treatment options.",
                    "Consider intra-articular injections, bracing, or other interventional therapies.",
                    "Surgical evaluation may be appropriate if conservative measures fail."
                ]
            },
            4: {
                "joint_space": "Severe joint space narrowing is present with greater than 50% reduction or near bone-on-bone contact, particularly in the medial compartment.",
                "osteophytes": "Large osteophytes are present throughout the joint, causing significant marginal irregularity.",
                "sclerosis": "Marked subchondral sclerosis is evident in all compartments.",
                "other": "Definite joint deformity with varus or valgus malalignment. Possible subluxation. Joint effusion may be present.",
                "conclusion": [
                    "Severe knee osteoarthritis (Kellgren-Lawrence Grade 4).",
                    "Advanced degenerative changes with marked joint space loss and deformity.",
                    "Urgent orthopedic referral recommended.",
                    "Surgical intervention (total knee arthroplasty) should be considered.",
                    "Quality of life assessment and surgical candidacy evaluation advised."
                ]
            }
        }
        
        grade_findings = findings_by_grade[kl_grade]
        
        # Build patient demographics
        demographics = []
        if patient_age:
            demographics.append(f"{patient_age}-year-old")
        if patient_sex:
            demographics.append(patient_sex)
        demographics.append("patient")
        demographics_text = " ".join(demographics)
        
        # Build clinical history
        if clinical_history:
            history_text = f"{demographics_text} with {clinical_history}."
        else:
            history_text = f"{demographics_text} presenting with knee pain and/or evaluation for osteoarthritis."
        
        # Extended information by grade
        extended_info = {
            0: {
                "characteristics": "Healthy knee joint with normal anatomical features. No degenerative changes visible on radiographic examination. Joint space is well-preserved with smooth articular surfaces.",
                "distribution": "Not applicable - no pathological changes present.",
                "risk": "Low risk for developing osteoarthritis. Maintain healthy lifestyle and regular physical activity.",
                "causes": ["Natural aging process (preventive stage)", "Maintaining joint health through appropriate exercise", "Genetic predisposition for future development"],
                "precautions": ["Continue regular physical activity and weight management", "Use proper form during exercise to prevent joint stress", "Consider preventive measures if family history of OA exists", "Annual check-ups for individuals over 50 years"],
                "treatments": ["No treatment required - maintain healthy lifestyle", "Continue regular low-impact exercise (swimming, cycling, walking)", "Weight management to prevent future joint stress", "Balanced diet rich in omega-3 fatty acids and antioxidants"]
            },
            1: {
                "characteristics": "Doubtful early osteoarthritic changes with minimal joint space narrowing. Questionable osteophyte formation. Represents earliest detectable changes that may or may not progress.",
                "distribution": "Changes may appear in medial or lateral compartments. Often asymmetric presentation. Most commonly affects weight-bearing regions of the joint.",
                "risk": "Moderate risk of progression to definite osteoarthritis. Risk factors include obesity, previous knee injury, family history, and high-impact activities.",
                "causes": ["Early degenerative changes due to aging", "Previous minor trauma or repetitive stress", "Genetic predisposition with advancing age", "Overweight or obesity increasing joint loading"],
                "precautions": ["Monitor symptoms regularly - report increased pain or stiffness", "Avoid high-impact activities (running, jumping, heavy lifting)", "Use proper footwear with good cushioning", "Consider knee-friendly exercises (swimming, cycling)", "Maintain ideal body weight to reduce joint stress"],
                "treatments": ["Observation with regular follow-up imaging (12-24 months)", "Physical therapy focusing on quadriceps strengthening", "Low-impact aerobic exercises", "Over-the-counter NSAIDs for occasional discomfort", "Glucosamine and chondroitin supplements (optional)", "Weight reduction if BMI > 25"]
            },
            2: {
                "characteristics": "Mild osteoarthritis with definite joint space narrowing and confirmed osteophyte formation. Subchondral bone changes beginning to develop. Represents established but early-stage disease.",
                "distribution": "Typically affects medial compartment more than lateral. May involve patellofemoral joint. Usually unilateral but can be bilateral with asymmetric severity.",
                "risk": "High risk of progression without intervention. Increased risk of developing moderate-to-severe OA within 5-10 years. Activity modification and treatment adherence critical.",
                "causes": ["Progressive cartilage degeneration", "Cumulative joint wear and tear from aging", "Previous knee injuries (meniscal tears, ligament injuries)", "Obesity (BMI > 30) significantly accelerates progression", "Occupational factors (prolonged standing, kneeling, squatting)"],
                "precautions": ["Avoid prolonged standing or walking on hard surfaces", "Use assistive devices (cane, walking stick) for long distances", "Apply ice after activities to reduce inflammation (15-20 minutes)", "Avoid stairs when possible - use ramps or elevators", "Wear supportive, cushioned footwear", "Consider knee braces for additional support during activity"],
                "treatments": ["Regular physical therapy (2-3 times per week)", "Prescription NSAIDs or COX-2 inhibitors", "Intra-articular corticosteroid injections (up to 3-4 per year)", "Hyaluronic acid (viscosupplementation) injections", "Topical analgesics (diclofenac gel, capsaicin cream)", "Weight loss program if overweight (target 5-10% body weight reduction)", "Low-impact exercises: swimming, water aerobics, stationary cycling"]
            },
            3: {
                "characteristics": "Moderate osteoarthritis with significant joint space narrowing (25-50% reduction), multiple osteophytes, and subchondral sclerosis. Joint deformity may be developing. Represents advanced degenerative disease.",
                "distribution": "Predominantly affects medial compartment with possible lateral involvement. Patellofemoral degenerative changes common. May show varus or valgus malalignment beginning to develop.",
                "risk": "Very high risk of progression to severe OA requiring surgery. Without intervention, likely progression to Grade 4 within 3-5 years. Quality of life significantly impacted.",
                "causes": ["Advanced cartilage loss with bone changes", "Long-standing osteoarthritis (typically 5-10+ years)", "Significant previous trauma (fractures, major ligament tears)", "Morbid obesity (BMI > 35)", "Genetic factors combined with environmental stressors", "Inflammatory conditions accelerating degeneration"],
                "precautions": ["Limit weight-bearing activities - use assistive devices regularly", "Avoid all high-impact activities completely", "Use bilateral support (walker) if both knees affected", "Apply heat before activity and ice after activity", "Sleep with pillow between knees if lying on side", "Consider occupational therapy for daily living adaptations", "Use elevated toilet seat and shower chair to reduce joint stress"],
                "treatments": ["Intensive physical therapy with aquatic therapy component", "Stronger prescription pain medications (tramadol, duloxetine)", "Regular intra-articular injections (every 3-4 months)", "PRP (Platelet-Rich Plasma) injections (experimental)", "Unloader knee braces to shift weight from affected compartment", "Surgical consultation for: osteotomy, partial knee replacement, or TKA evaluation", "Pain management specialist referral", "Weight loss program mandatory if BMI > 30"]
            },
            4: {
                "characteristics": "Severe osteoarthritis with marked joint space loss (>50% or bone-on-bone contact), large osteophytes throughout, extensive subchondral sclerosis, and definite joint deformity. End-stage degenerative joint disease.",
                "distribution": "Involves all compartments of the knee (tricompartmental disease). Severe varus or valgus deformity present. Possible subluxation. Often bilateral involvement though asymmetric.",
                "risk": "End-stage disease - surgical intervention typically necessary for functional improvement. Non-surgical management provides only limited symptomatic relief. Significant disability without treatment.",
                "causes": ["Long-standing severe osteoarthritis (typically 10+ years)", "Complete cartilage loss with bone remodeling", "Previous major trauma, fractures, or failed surgeries", "Severe obesity (BMI > 40) or long-term morbid obesity", "Genetic predisposition with multiple risk factors", "Inflammatory arthropathies (rheumatoid arthritis, post-traumatic arthritis)"],
                "precautions": ["Minimize all weight-bearing activities", "Use assistive devices for all ambulation (walker, wheelchair for long distances)", "Fall prevention measures - remove tripping hazards at home", "Use adaptive equipment for daily activities", "Avoid stairs entirely - consider stair lift or bedroom relocation", "Pain management critical - use prescribed medications as directed", "Monitor for complications: depression, deconditioning, weight gain"],
                "treatments": ["Urgent orthopedic referral for total knee arthroplasty (TKA) evaluation", "Pre-surgical optimization: cardiac clearance, diabetes control, smoking cessation", "Pain management: opioid alternatives, nerve blocks, radiofrequency ablation", "Maximal medical therapy: NSAIDs, acetaminophen, topical agents", "Final attempts at injection therapy if surgery not possible", "Comprehensive pre-hab physical therapy program", "Nutritional counseling and mandatory weight optimization", "Psychological support for chronic pain and surgical preparation", "Disability evaluation and work accommodations if applicable"]
            }
        }
        
        info = extended_info[kl_grade]
        
        # Construct comprehensive report
        report = f"""DISCLAIMER:
This report is generated using artificial intelligence analysis and should be considered preliminary. Definitive diagnosis requires in-person evaluation by a board-certified radiologist or orthopedic specialist. Please consult a healthcare professional for treatment planning.

CLINICAL HISTORY:
{history_text}

TECHNIQUE:
Standard anteroposterior (AP) and lateral radiographic views of the knee were obtained and analyzed using deep learning-based AI classification system (ResNet50 architecture) trained on validated knee osteoarthritis datasets.
AI Classification Confidence: {confidence:.1%}
Kellgren-Lawrence Grading System Applied

FINDINGS:
Joint Space: {grade_findings['joint_space']}

Osteophytes: {grade_findings['osteophytes']}

Subchondral Changes: {grade_findings['sclerosis']}

Additional Observations: {grade_findings['other']}

DISEASE CHARACTERISTICS:
{info['characteristics']}

ANATOMICAL DISTRIBUTION:
{info['distribution']}

RISK PROFILE:
{info['risk']}

MAIN CAUSES:
"""
        for cause in info['causes']:
            report += f"• {cause}\n"
        
        report += f"""
CLINICAL RECOMMENDATIONS:
"""
        for i, conclusion in enumerate(grade_findings['conclusion'], 1):
            report += f"{i}. {conclusion}\n"
        
        report += f"""
ESSENTIAL PRECAUTIONS:
"""
        for precaution in info['precautions']:
            report += f"• {precaution}\n"
        
        report += f"""
TREATMENT OPTIONS:
"""
        for treatment in info['treatments']:
            report += f"• {treatment}\n"
        
        report += f"""
PATHOPHYSIOLOGY:
Knee osteoarthritis is a degenerative joint disease characterized by progressive loss of articular cartilage, subchondral bone changes, and osteophyte formation. The pathological process involves mechanical stress, inflammatory mediators, and enzymatic degradation of cartilage matrix. Risk factors include aging (most significant), obesity, previous joint injury, genetic predisposition, occupational stress, and biomechanical abnormalities. The Kellgren-Lawrence grading system (Grade 0-4) provides standardized assessment of radiographic severity.

---
Generated by Advanced AI Knee Osteoarthritis Analysis System
"""
        
        return report.strip()
    
    def translate_report(self, report_text, target_language='en'):
        """
        Translate report to target language with multiple fallbacks:
        1. Try googletrans (free, fast)
        2. Try Gemini API (free tier, reliable)
        3. Return English report if both fail
        """
        if target_language == 'en' or target_language is None:
            return report_text
        
        # Language name mapping for better Gemini prompts
        language_names = {
            'es': 'Spanish', 'fr': 'French', 'de': 'German', 'hi': 'Hindi',
            'ta': 'Tamil', 'te': 'Telugu', 'kn': 'Kannada', 'ml': 'Malayalam',
            'zh': 'Chinese', 'ja': 'Japanese', 'ar': 'Arabic', 'pt': 'Portuguese',
            'ru': 'Russian', 'ko': 'Korean', 'it': 'Italian', 'nl': 'Dutch',
            'bn': 'Bengali', 'mr': 'Marathi', 'gu': 'Gujarati', 'pa': 'Punjabi'
        }
        language_name = language_names.get(target_language, target_language)
        
        # Try Method 1: deep-translator (fast, free)
        if self.translator:
            try:
                print(f"🔄 Translating report to {language_name} using GoogleTranslator...")
                translator = GoogleTranslator(source='auto', target=target_language)
                translated = translator.translate(report_text)
                print("✅ Translation completed with GoogleTranslator")
                return translated
            except Exception as e:
                print(f"⚠️ GoogleTranslator failed: {e}")
                print("🔄 Trying Gemini API fallback...")
        
        # Try Method 2: Gemini API (more reliable)
        if self.gemini_model:
            try:
                print(f"🔄 Translating report to {language_name} using Gemini API...")
                prompt = f"""Translate the following medical radiology report to {language_name}.
Maintain all medical terminology accuracy and formatting.
Keep the report professional and clinically appropriate.

Report to translate:
{report_text}

Provide ONLY the translated text, no explanations or additional text."""
                
                response = self.gemini_model.generate_content(prompt)
                translated_text = response.text.strip()
                print(f"✅ Translation completed with Gemini API")
                return translated_text
            except Exception as e:
                print(f"⚠️ Gemini API translation failed: {e}")
        
        # Fallback: Return English report
        print(f"⚠️ All translation methods failed, returning English report")
        return report_text

    def generate_complete_report(self, kl_grade, confidence, language='en',
                                 patient_age=None, patient_sex=None, clinical_history=None):
        """
        Complete workflow: Generate English report with Med-Gemma, then translate
        
        Returns:
            tuple: (english_report, translated_report)
        """
        # Step 1: Generate English report with Med-Gemma
        print(f"Generating radiology report for KL Grade {kl_grade}...")
        english_report = self.generate_report_with_medgemma(
            kl_grade, confidence, patient_age, patient_sex, clinical_history
        )
        
        # Step 2: Translate if needed
        if language != 'en':
            translated_report = self.translate_report(english_report, language)
        else:
            translated_report = english_report
        
        return english_report, translated_report


# Main function for easy integration
def generate_radiology_report(kl_grade, confidence, language='en',
                              patient_age=None, patient_sex=None, clinical_history=None):
    """
    Main function to generate radiology report
    
    Args:
        kl_grade: Kellgren-Lawrence grade (0-4)
        confidence: Model confidence (0-1)
        language: Target language code (default: 'en')
        patient_age: Patient age in years
        patient_sex: Patient sex ('Male'/'Female')
        clinical_history: Clinical history text
    
    Returns:
        tuple: (english_report, translated_report)
    """
    generator = RadiologyReportGenerator()
    return generator.generate_complete_report(
        kl_grade, confidence, language, patient_age, patient_sex, clinical_history
    )


if __name__ == '__main__':
    # Test report generation
    print("Testing Radiology Report Generator")
    print("=" * 60)
    
    # Test with different KL grades
    for grade in [0, 2, 4]:
        print(f"\nTesting KL Grade {grade}:")
        english, translated = generate_radiology_report(
            kl_grade=grade,
            confidence=0.87,
            language='en',
            patient_age=65,
            patient_sex='Male',
            clinical_history='knee pain for 6 months'
        )
        print(english[:200] + "...")

