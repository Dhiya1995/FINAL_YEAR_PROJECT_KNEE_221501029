import os
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google-generativeai not available.")

from config import GEMINI_API_KEY


class PrescriptiveReportGenerator:
    
    def __init__(self):
        self.model = None
        
        if GENAI_AVAILABLE and GEMINI_API_KEY:
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
                    self.model = genai.GenerativeModel(model_name)
                    print(f"✅ Gemini configured: {model_name}")
                    break
                except Exception as e:
                    print(f"⚠️ Model {model_name} unavailable: {e}")
                    continue
            
            if not self.model:
                print("⚠️ All Gemini models failed")
        else:
            print("⚠️ Gemini not available")
    
    def _calculate_bmi(self, height_cm, weight_kg):
        if not height_cm or not weight_kg:
            return None
        try:
            height_m = height_cm / 100
            bmi = weight_kg / (height_m ** 2)
            return round(bmi, 1)
        except:
            return None
    
    def _get_bmi_category(self, bmi):
        if bmi is None:
            return "Unknown"
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal weight"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    def _create_comprehensive_prompt(self, kl_grade, height, weight, age, activity_level, sex=None):
        
        bmi = self._calculate_bmi(height, weight)
        bmi_category = self._get_bmi_category(bmi)
        
        kl_info = {
            0: {"name": "No Osteoarthritis", "severity": "None", "focus": "prevention"},
            1: {"name": "Doubtful OA", "severity": "Minimal", "focus": "early intervention"},
            2: {"name": "Mild OA", "severity": "Mild", "focus": "symptom management"},
            3: {"name": "Moderate OA", "severity": "Moderate", "focus": "comprehensive treatment"},
            4: {"name": "Severe OA", "severity": "Severe", "focus": "advanced treatment options"}
        }
        
        grade_info = kl_info[kl_grade]
        
        patient_summary = f"{age}-year-old {sex if sex else ''} patient".strip()
        
        prompt = f"""You are an expert orthopedic specialist and physical therapist. Generate a COMPLETE, detailed, personalized prescriptive care plan for knee osteoarthritis.

PATIENT INFORMATION:
- Age: {age} years
- Sex: {sex if sex else 'Not specified'}
- Height: {height} cm
- Weight: {weight} kg
- BMI: {bmi} ({bmi_category})
- Activity Level: {activity_level}

DIAGNOSIS:
- Kellgren-Lawrence Grade: {kl_grade}
- Severity: {grade_info['severity']}
- Diagnosis: {grade_info['name']}
- Treatment Focus: {grade_info['focus']}

Generate a COMPLETE prescriptive care plan with ALL sections below. Write detailed, specific, actionable medical content for each section:

═══════════════════════════════════════════════════════════

1. OVERVIEW
Write 3-4 sentences summarizing the patient's condition, prognosis, and treatment goals specific to KL Grade {kl_grade}, considering age {age} and BMI {bmi}.

2. WEIGHT MANAGEMENT
Provide specific weight management recommendations based on BMI {bmi} ({bmi_category}):
- Target weight if applicable (calculate based on BMI)
- Timeline and realistic weight loss goals (if needed)
- Strategies for achieving/maintaining healthy weight
- Impact of weight on knee osteoarthritis Grade {kl_grade}
Write 4-5 detailed sentences with specific targets.

3. EXERCISE AND PHYSICAL THERAPY
Provide comprehensive exercise plan appropriate for:
- Age: {age} years
- Activity level: {activity_level}
- KL Grade: {kl_grade}

Include:
- Specific exercises with frequency and duration
- Progression plan over weeks
- Exercises to AVOID
- Physical therapy recommendations
- Expected benefits
Write 6-8 detailed bullet points or paragraphs.

4. PAIN MANAGEMENT
Provide detailed pain management strategies for KL Grade {kl_grade}:
- Medication options (OTC and prescription)
- Dosing recommendations
- Non-pharmacological approaches (ice, heat, TENS, etc.)
- Advanced interventions if applicable (injections, etc.)
- When to escalate care
- Side effects to monitor
Write 5-7 specific recommendations.

5. LIFESTYLE MODIFICATIONS
Provide specific daily activity modifications for KL Grade {kl_grade}:
- Home modifications
- Work adaptations
- Activity pacing strategies
- Sleep and rest recommendations
- Assistive devices if needed
- Ergonomic adjustments
Write 5-6 detailed recommendations.

6. NUTRITION AND SUPPLEMENTS
Provide anti-inflammatory nutrition plan:
- Specific foods to include (with quantities)
- Foods to avoid
- Supplement recommendations (glucosamine, omega-3, etc.) with dosages
- Hydration guidelines
- Meal planning tips
Write 5-6 detailed recommendations.

7. ACTIVITY GUIDELINES
Specify safe vs. unsafe activities for KL Grade {kl_grade}:
- RECOMMENDED activities (list 5-6)
- Activities to MODIFY (list 3-4 with modifications)
- Activities to AVOID (list 4-5)
- Sports and recreation guidance
- Daily task adaptations

8. ADVANCED TREATMENT OPTIONS
Describe treatment options appropriate for KL Grade {kl_grade}:
- Conservative treatments still applicable
- Injection therapies (corticosteroid, HA, PRP)
- Surgical options if applicable
- When to consider each option
- Referral recommendations
Write 5-7 specific treatment modalities.

9. FOLLOW-UP AND MONITORING
Specify monitoring plan for KL Grade {kl_grade}:
- Follow-up schedule (specific timeframes)
- Imaging frequency
- Warning signs to monitor
- When to escalate care
- Expected outcomes and timeline
- Quality of life assessments
Write 4-5 specific monitoring guidelines.

10. RESOURCES AND SUPPORT
Provide helpful resources:
- Educational websites and organizations
- Specialist referrals needed
- Support groups
- Mobile apps or tools
- Community resources
Write 4-5 specific resources.

═══════════════════════════════════════════════════════════

IMPORTANT:
- Make ALL recommendations specific to this patient's profile (age {age}, BMI {bmi}, activity level {activity_level}, KL Grade {kl_grade})
- Use evidence-based medical guidelines
- Be detailed and actionable - give specific exercises, dosages, timeframes
- Write in professional but compassionate tone
- Make it comprehensive - this should be a complete care plan

Generate the complete prescriptive care plan now:"""
        
        return prompt
    
    def _generate_with_gemini(self, kl_grade, height, weight, age, activity_level, sex=None):
        if not self.model:
            return None
        
        try:
            print("🤖 Generating prescriptive care plan with Gemini...")
            prompt = self._create_comprehensive_prompt(
                kl_grade, height, weight, age, activity_level, sex
            )
            
            generation_config = {
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40,
                'max_output_tokens': 3072,
            }
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            report = response.text.strip()
            
            if len(report) > 1000:
                print(f"✅ Gemini generated {len(report)} chars")
                return report
            else:
                print("⚠️ Gemini report too short")
                return None
                
        except Exception as e:
            print(f"⚠️ Gemini generation error: {e}")
            return None
    
    def _generate_minimal_fallback(self, kl_grade, height, weight, age, activity_level, sex=None):
        print("⚠️ GEMINI FAILED - Using minimal emergency fallback")
        
        bmi = self._calculate_bmi(height, weight)
        bmi_category = self._get_bmi_category(bmi)
        
        kl_names = {
            0: "No Osteoarthritis",
            1: "Doubtful Osteoarthritis",
            2: "Mild Osteoarthritis",
            3: "Moderate Osteoarthritis",
            4: "Severe Osteoarthritis"
        }
        
        return f"""PRESCRIPTIVE CARE PLAN
Knee Osteoarthritis Management

═══════════════════════════════════════════════════════════

PATIENT INFORMATION
─────────────────────────────────────────────────────────────
{age}-year-old {sex if sex else ''} patient
Height: {height} cm
Weight: {weight} kg
BMI: {bmi} ({bmi_category})
Activity Level: {activity_level.capitalize()}
Diagnosis: Kellgren-Lawrence Grade {kl_grade} - {kl_names.get(kl_grade, "OA")}

═══════════════════════════════════════════════════════════

1. OVERVIEW
You have been diagnosed with {kl_names.get(kl_grade, "knee osteoarthritis")}. A comprehensive treatment plan combining exercise, weight management, and appropriate medical interventions can help manage symptoms and improve function.

2. WEIGHT MANAGEMENT
{"Weight management is recommended. Consult with healthcare provider for personalized weight loss plan." if bmi and bmi > 25 else "Maintain current healthy weight through balanced diet and regular exercise."}

3. EXERCISE AND PHYSICAL THERAPY
• Low-impact aerobic exercise 30 minutes, 3-5 times per week
• Strengthening exercises for quadriceps and hamstrings
• Range of motion exercises daily
• Physical therapy consultation recommended
• Avoid high-impact activities

4. PAIN MANAGEMENT
• Over-the-counter pain relievers as needed (acetaminophen, NSAIDs)
• Ice after activities (15-20 minutes)
• Heat before exercise
• Consult physician for persistent pain

5. LIFESTYLE MODIFICATIONS
• Pace activities throughout the day
• Use assistive devices if needed
• Modify work environment for comfort
• Ensure adequate rest and sleep

6. NUTRITION AND SUPPLEMENTS
• Anti-inflammatory diet rich in fruits, vegetables, and omega-3
• Stay well-hydrated
• Consider glucosamine/chondroitin supplements (consult physician)
• Maintain balanced nutrition

7. ACTIVITY GUIDELINES
RECOMMENDED: Swimming, cycling, walking, water aerobics
AVOID: Running, jumping, high-impact sports
MODIFY: Daily activities to reduce knee stress

8. ADVANCED TREATMENT OPTIONS
{"Consult orthopedic surgeon for surgical evaluation options." if kl_grade >= 3 else "Conservative management with possible injections if symptoms persist."}
Discuss with healthcare provider about:
• Corticosteroid injections
• Hyaluronic acid injections
• Physical therapy
{"• Surgical options (partial or total knee replacement)" if kl_grade >= 3 else ""}

9. FOLLOW-UP AND MONITORING
• Follow-up with healthcare provider every {3 if kl_grade >= 3 else 6} months
• Monitor pain levels and functional status
• Imaging as clinically indicated
• Report worsening symptoms promptly

10. RESOURCES AND SUPPORT
• Arthritis Foundation (www.arthritis.org)
• American Academy of Orthopaedic Surgeons (www.aaos.org)
• Physical therapist for exercise guidance
• Orthopedic specialist consultation as needed

═══════════════════════════════════════════════════════════

IMPORTANT: This is a basic care plan. Consult with healthcare professionals for personalized treatment recommendations.

Generated: {age}-year-old {sex if sex else 'patient'} | KL Grade {kl_grade} | BMI: {bmi}"""
    
    def generate_report(self, kl_grade, height, weight, age, activity_level, sex=None):
        """
        Main report generation - tries Gemini, minimal fallback only if fails
        
        Priority:
        1. Gemini API (best for comprehensive generation)
        2. Minimal emergency fallback
        """
        # Try Gemini first
        report = self._generate_with_gemini(
            kl_grade, height, weight, age, activity_level, sex
        )
        if report:
            return report
        
        # Emergency fallback
        print("⚠️ Gemini generation failed! Using minimal fallback")
        return self._generate_minimal_fallback(
            kl_grade, height, weight, age, activity_level, sex
        )
    
    def generate_complete_report(self, kl_grade, height, weight, age, activity_level, sex=None):
        """
        Main entry point for report generation
        """
        print(f"📝 Generating prescriptive care plan for KL Grade {kl_grade}...")
        
        report = self.generate_report(
            kl_grade, height, weight, age, activity_level, sex
        )
        
        return report


def generate_prescriptive_report(kl_grade, height, weight, age, activity_level, sex=None):
    generator = PrescriptiveReportGenerator()
    return generator.generate_complete_report(
        kl_grade, height, weight, age, activity_level, sex
    )


if __name__ == '__main__':
    print("Testing LLM-First Prescriptive Report Generator")
    print("=" * 60)
    
    report = generate_prescriptive_report(
        kl_grade=2,
        height=175,
        weight=85,
        age=55,
        activity_level='moderate',
        sex='Male'
    )
    
    print(report[:500] + "...")
