
import os
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google-generativeai not available. Using template-based generation.")

from config import GEMINI_API_KEY


class PrescriptiveReportGenerator:
    """
    Generate prescriptive care and treatment recommendations using Gemini API
    
    Workflow:
    1. Structured Prompt Creation: Based on patient details (height, weight, age, activity level) and KL grade
    2. Gemini API Integration: Generate personalized care plan
    3. Output: Structured prescriptive care report
    """
    
    def __init__(self):
        self.model = None
        
        if GENAI_AVAILABLE and GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            # Try multiple Gemini model versions for compatibility
            gemini_models = [
                'gemini-2.0-flash-exp',       # ✅ WORKS with your API key!
                'gemini-1.5-flash-latest',    # Fallback options below
                'gemini-1.5-flash-002',       
                'gemini-1.5-flash',           
                'gemini-1.5-pro-latest',      
                'gemini-pro',                 
            ]
            
            for model_name in gemini_models:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    # Don't test during initialization - just create the model object
                    print(f"✅ Gemini API configured with model: {model_name}")
                    break
                except Exception as e:
                    print(f"⚠️ Model {model_name} not available: {e}")
                    continue
            
            if not self.model:
                print("⚠️ All Gemini models failed to initialize. Using template-based generation.")
        else:
            if not GENAI_AVAILABLE:
                print("⚠️ google-generativeai library not installed")
            if not GEMINI_API_KEY:
                print("⚠️ GEMINI_API_KEY not set in environment variables")
            print("Using template-based prescriptive report generation")
    
    def _calculate_bmi(self, height_cm, weight_kg):
        """Calculate BMI from height and weight"""
        if not height_cm or not weight_kg:
            return None
        try:
            height_m = height_cm / 100
            bmi = weight_kg / (height_m ** 2)
            return round(bmi, 1)
        except:
            return None
    
    def _get_bmi_category(self, bmi):
        """Get BMI category"""
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
    
    def _create_structured_prompt(self, kl_grade, height, weight, age, activity_level, sex=None):
        """
        Create structured prompt for Gemini API
        Based on patient details and classified KL grade
        """
        
        bmi = self._calculate_bmi(height, weight)
        bmi_category = self._get_bmi_category(bmi)
        
        # KL Grade descriptions and severity
        kl_info = {
            0: {
                "name": "No Osteoarthritis",
                "severity": "None",
                "focus": "prevention and maintaining joint health"
            },
            1: {
                "name": "Doubtful/Minimal Osteoarthritis",
                "severity": "Minimal",
                "focus": "early intervention and preventing progression"
            },
            2: {
                "name": "Mild Osteoarthritis",
                "severity": "Mild",
                "focus": "symptom management and slowing disease progression"
            },
            3: {
                "name": "Moderate Osteoarthritis",
                "severity": "Moderate",
                "focus": "comprehensive pain management and functional improvement"
            },
            4: {
                "name": "Severe Osteoarthritis",
                "severity": "Severe",
                "focus": "advanced treatment options and quality of life improvement"
            }
        }
        
        grade_info = kl_info[kl_grade]
        
        # Build patient profile summary
        patient_summary = []
        if age:
            patient_summary.append(f"{age}-year-old")
        if sex:
            patient_summary.append(sex)
        patient_summary.append("patient")
        patient_summary_text = " ".join(patient_summary)
        
        # Create comprehensive prompt
        prompt = f"""You are an expert orthopedic specialist and physical therapist. Generate a comprehensive, personalized prescriptive care plan for knee osteoarthritis.

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

PRIMARY FOCUS: The treatment plan should focus on {grade_info['focus']}.

Please generate a detailed, personalized care plan with the following sections:

1. OVERVIEW
   - Brief summary of the patient's condition and treatment goals

2. WEIGHT MANAGEMENT
   - Specific recommendations based on BMI ({bmi}, {bmi_category})
   - Target weight if applicable
   - Timeline and realistic goals

3. EXERCISE AND PHYSICAL THERAPY
   - Specific exercises appropriate for age ({age}) and activity level ({activity_level})
   - Frequency, duration, and intensity guidelines
   - Exercises to avoid
   - Progression plan

4. PAIN MANAGEMENT
   - Appropriate pain management strategies for KL Grade {kl_grade}
   - Medication recommendations (OTC and prescription considerations)
   - Non-pharmacological approaches (ice, heat, elevation, etc.)
   - When to seek medical attention

5. LIFESTYLE MODIFICATIONS
   - Daily activity modifications
   - Ergonomic adjustments
   - Assistive devices if needed
   - Sleep and rest recommendations

6. NUTRITION AND SUPPLEMENTS
   - Anti-inflammatory diet recommendations
   - Specific foods to include and avoid
   - Supplement considerations (glucosamine, chondroitin, omega-3, etc.)
   - Hydration guidelines

7. ACTIVITY GUIDELINES
   - Safe activities and sports
   - Activities to modify or avoid
   - Adaptive strategies for daily tasks
   - Work-related modifications if applicable

8. ADVANCED TREATMENT OPTIONS
   - When conservative management may not be sufficient
   - Injections (corticosteroid, hyaluronic acid, PRP)
   - Surgical options for KL Grade {kl_grade}
   - Referral recommendations

9. FOLLOW-UP AND MONITORING
   - Recommended follow-up schedule
   - Warning signs to monitor
   - When to escalate care
   - Expected outcomes and timeline

10. RESOURCES AND SUPPORT
    - Recommended resources for patient education
    - Support groups or community resources
    - Physical therapy and specialist referrals

Make the recommendations:
- Specific and actionable
- Evidence-based
- Tailored to the patient's age, weight, activity level, and KL grade
- Realistic and achievable
- Clear about potential risks and benefits

Generate a professional, compassionate, and comprehensive care plan:"""
        
        return prompt
    
    def generate_report_with_gemini(self, kl_grade, height, weight, age, activity_level, sex=None):
        """
        Generate prescriptive care report using Gemini API
        """
        
        prompt = self._create_structured_prompt(kl_grade, height, weight, age, activity_level, sex)
        
        if self.model is None:
            print("Gemini API not available, using template...")
            return self._generate_template_report(kl_grade, height, weight, age, activity_level, sex)
        
        try:
            print("Generating prescriptive care plan with Gemini API...")
            
            # Configure generation parameters
            generation_config = {
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40,
                'max_output_tokens': 2048,
            }
            
            # Generate content with Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            report = response.text.strip()
            
            # Validate report length
            if len(report) < 200:
                print("Generated report too short, using template...")
                return self._generate_template_report(kl_grade, height, weight, age, activity_level, sex)
            
            print("✅ Prescriptive care plan generated successfully with Gemini")
            return report
            
        except Exception as e:
            print(f"⚠️ Error generating report with Gemini: {e}")
            print("Falling back to template-based generation...")
            return self._generate_template_report(kl_grade, height, weight, age, activity_level, sex)
    
    def _generate_template_report(self, kl_grade, height, weight, age, activity_level, sex=None):
        """
        Fallback template-based report generation
        Ensures report is always generated
        """
        
        bmi = self._calculate_bmi(height, weight)
        bmi_category = self._get_bmi_category(bmi)
        
        # Build patient demographics
        demographics = []
        if age:
            demographics.append(f"{age}-year-old")
        if sex:
            demographics.append(sex)
        demographics_text = " ".join(demographics) + " patient" if demographics else "Patient"
        
        report = f"""PRESCRIPTIVE CARE PLAN
Knee Osteoarthritis Management

═══════════════════════════════════════════════════════════

PATIENT INFORMATION
─────────────────────────────────────────────────────────────
{demographics_text}
Height: {height} cm
Weight: {weight} kg
BMI: {bmi} ({bmi_category})
Activity Level: {activity_level.capitalize()}
Kellgren-Lawrence Grade: {kl_grade}

═══════════════════════════════════════════════════════════

1. OVERVIEW
─────────────────────────────────────────────────────────────
"""
        
        # Grade-specific overview
        if kl_grade == 0:
            report += """Your knee X-rays show no signs of osteoarthritis. The focus of this care plan 
is on prevention and maintaining healthy joints through lifestyle modifications 
and appropriate exercise."""
        elif kl_grade == 1:
            report += """Your knee X-rays show early/doubtful signs of osteoarthritis. Early intervention 
now can help slow disease progression and maintain joint function. The focus is on 
strengthening, weight management (if applicable), and activity modification."""
        elif kl_grade == 2:
            report += """Your knee X-rays show mild osteoarthritis with definite joint space narrowing 
and osteophytes. Conservative management with exercise, weight control, and pain 
management strategies can significantly improve symptoms and function."""
        elif kl_grade == 3:
            report += """Your knee X-rays show moderate osteoarthritis with significant joint changes. 
Comprehensive treatment combining physical therapy, weight management, pain control, 
and possibly advanced interventions will be important for managing symptoms and 
maintaining function."""
        else:  # Grade 4
            report += """Your knee X-rays show severe osteoarthritis with advanced degenerative changes. 
While conservative management remains important, you may be a candidate for 
surgical intervention. This plan includes both conservative measures and guidance 
on when to consider advanced treatment options."""
        
        report += "\n\n2. WEIGHT MANAGEMENT\n─────────────────────────────────────────────────────────────\n"
        
        if bmi and bmi > 25:
            weight_to_lose = round((bmi - 24.9) * ((height / 100) ** 2), 1)
            report += f"""Your current BMI is {bmi} ({bmi_category}). Weight management is crucial for 
reducing stress on knee joints.

• Target: Lose {weight_to_lose} kg to reach healthy BMI range (18.5-24.9)
• Goal: Aim for 0.5-1 kg per week weight loss (safe and sustainable)
• Strategy: Combine caloric deficit (500 calories/day) with increased activity
• Benefits: Each kg lost reduces knee joint load by approximately 4 kg
• Timeline: {int(weight_to_lose * 4)} months to reach target weight
"""
        elif bmi and bmi < 18.5:
            report += f"""Your current BMI is {bmi} (Underweight). Focus on maintaining adequate nutrition 
to support joint health and muscle strength.

• Ensure adequate protein intake (1.2-1.5 g per kg body weight)
• Include anti-inflammatory foods in diet
• Consult with nutritionist if needed
"""
        else:
            report += f"""Your current BMI is {bmi} (Normal range). Maintain current weight through:

• Balanced diet with anti-inflammatory foods
• Regular physical activity
• Monitor weight monthly
• Adjust if BMI increases above 25
"""
        
        report += "\n\n3. EXERCISE AND PHYSICAL THERAPY\n─────────────────────────────────────────────────────────────\n"
        
        if activity_level.lower() in ['sedentary', 'low', 'minimal']:
            report += """Start with low-intensity exercises and gradually increase:

• Week 1-2: Range of motion exercises (10 min daily)
  - Gentle knee bends, leg raises
  - Ankle pumps, heel slides
  
• Week 3-4: Add strengthening exercises (15 min, 3x/week)
  - Quad sets, straight leg raises
  - Hamstring curls, calf raises
  
• Week 5+: Add low-impact aerobic exercise (20-30 min, 3-4x/week)
  - Walking, swimming, or stationary cycling
  - Water aerobics (excellent for joint protection)

"""
        else:
            report += """Maintain active lifestyle with modifications:

• Continue low-impact aerobic activities (30 min, 4-5x/week)
  - Walking, swimming, cycling, elliptical
  - Avoid running and jumping if painful
  
• Strength training (20-30 min, 2-3x/week)
  - Focus on quadriceps, hamstrings, hip muscles
  - Resistance bands or light weights
  - Body weight exercises
  
"""
        
        if kl_grade >= 2:
            report += """• Physical therapy consultation HIGHLY RECOMMENDED
  - Personalized exercise program
  - Manual therapy techniques
  - Gait training and biomechanical assessment
  - Expected: 6-12 sessions over 2-3 months

"""
        
        report += """• Exercises to AVOID:
  - High-impact activities (running, jumping)
  - Deep squatting or lunging
  - Kneeling or prolonged standing
  - Activities causing sharp pain

"""
        
        report += "\n\n4. PAIN MANAGEMENT\n─────────────────────────────────────────────────────────────\n"
        
        if kl_grade <= 1:
            report += """Conservative pain management approach:

• Over-the-counter options:
  - Acetaminophen (Tylenol): 500-1000 mg every 6 hours as needed
  - Topical NSAIDs (diclofenac gel): Apply to affected area 2-4x daily
  
• Non-pharmacological methods:
  - Ice: 15-20 minutes after activity or when swollen
  - Heat: 15-20 minutes before exercise (NOT if inflamed/swollen)
  - Compression: Knee sleeve for support during activities
  
• When to escalate:
  - If pain persists despite 6 weeks of conservative management
  - If pain significantly impacts daily activities or sleep
"""
        
        elif kl_grade == 2:
            report += """Multi-modal pain management approach:

• First-line medications:
  - Acetaminophen: 500-1000 mg every 6 hours
  - Oral NSAIDs: Ibuprofen 400-600 mg every 8 hours with food
    (Use lowest effective dose, limit to 7-10 days if possible)
  - Topical NSAIDs: Daily application to knee
  
• Alternative therapies:
  - Transcutaneous electrical nerve stimulation (TENS)
  - Acupuncture (may provide relief for some patients)
  - Massage therapy
  
• Bracing/Support:
  - Knee sleeve or brace for activities
  - Consider unloader brace if medial compartment affected
  
• When to see doctor:
  - Persistent pain despite medication
  - Need for regular NSAID use beyond 2 weeks
  - Consider intra-articular injections
"""
        
        else:  # Grades 3-4
            report += """Comprehensive pain management approach:

• Medications (consult with physician):
  - NSAIDs: Regular use may be necessary (monitor side effects)
  - Consider prescription options if OTC insufficient
  - Topical treatments: Capsaicin cream, lidocaine patches
  
• Advanced interventions to discuss with doctor:
  - Corticosteroid injections: For acute flares (3-4x per year max)
  - Hyaluronic acid injections: Series of 3-5 injections
  - PRP (Platelet-Rich Plasma): Emerging treatment option
  
• Non-pharmacological:
  - TENS unit: Daily use as needed
  - Heat/ice therapy: Alternate based on symptoms
  - Compression and elevation after activities
  
• Assistive devices:
  - Walking aids (cane, walker) to reduce joint load
  - Knee brace or unloader brace
  - Proper footwear with cushioning
  
• Red flags - seek immediate care if:
  - Sudden, severe pain or swelling
  - Unable to bear weight
  - Fever with knee pain (possible infection)
  - Locking or giving way of knee
"""
        
        report += "\n\n5. LIFESTYLE MODIFICATIONS\n─────────────────────────────────────────────────────────────\n"
        
        report += """Daily activity adjustments to protect your knees:

• Home modifications:
  - Use elevated toilet seat if difficulty rising
  - Install grab bars in bathroom
  - Minimize stair use (live on one level if possible)
  - Use sturdy chair with arms for easier rising
  
• Work considerations:
  - Avoid prolonged standing (use anti-fatigue mat if needed)
  - Take frequent breaks (every 30-60 minutes)
  - Avoid repetitive kneeling or squatting
  - Request ergonomic assessment if needed
  
• Activity modifications:
  - Pace activities (don't overdo on "good days")
  - Alternate sitting and standing
  - Use proper body mechanics when lifting
  - Avoid activities requiring deep bending
  
• Sleep and rest:
  - Sleep 7-9 hours per night for healing
  - Place pillow between knees if side sleeper
  - Elevate legs periodically during day
  - Rest knee after prolonged activity
"""
        
        report += "\n\n6. NUTRITION AND SUPPLEMENTS\n─────────────────────────────────────────────────────────────\n"
        
        report += """Anti-inflammatory diet for joint health:

• Foods to INCLUDE:
  - Fatty fish (salmon, mackerel, sardines): 2-3x per week
  - Colorful fruits and vegetables: 5-9 servings daily
  - Nuts and seeds (especially walnuts, flaxseeds)
  - Olive oil, avocado (healthy fats)
  - Green tea, turmeric, ginger
  - Whole grains instead of refined grains
  
• Foods to LIMIT or AVOID:
  - Processed and fried foods
  - Red meat and processed meats
  - Sugary beverages and desserts
  - Excessive alcohol
  - High-sodium foods
  
• Supplements to consider (consult physician first):
  - Glucosamine sulfate: 1500 mg daily
  - Chondroitin sulfate: 800-1200 mg daily
  - Omega-3 fatty acids: 2-3 grams daily (fish oil)
  - Vitamin D: Maintain levels >30 ng/mL
  - Curcumin/turmeric: 500-1000 mg daily
  
• Hydration:
  - Drink 8-10 glasses of water daily
  - Adequate hydration supports joint lubrication
"""
        
        report += "\n\n7. ACTIVITY GUIDELINES\n─────────────────────────────────────────────────────────────\n"
        
        report += """Safe activities and sports recommendations:

• RECOMMENDED activities:
  - Swimming and water aerobics (excellent - buoyancy reduces load)
  - Cycling (stationary or outdoor on flat terrain)
  - Walking on level surfaces
  - Elliptical machine
  - Tai chi or gentle yoga
  - Golf (with cart)
  
• MODIFY these activities:
  - Tennis/pickleball: Play doubles, avoid sudden pivots
  - Dancing: Avoid movements requiring deep knee bending
  - Gardening: Use kneeling pad, take frequent breaks
  - Hiking: Use trekking poles, stick to flat/gentle trails
  
• AVOID these activities:
  - Running or jogging
  - High-impact aerobics
  - Basketball, volleyball
  - Skiing (downhill)
  - Activities requiring jumping or sudden direction changes
  
• Daily activity tips:
  - Take stairs slowly, one step at a time if needed
  - Use handrails
  - Carry lighter loads (multiple trips vs. heavy load)
  - Sit to dress lower body
  - Use reaching aids for items on floor
"""
        
        report += "\n\n8. ADVANCED TREATMENT OPTIONS\n─────────────────────────────────────────────────────────────\n"
        
        if kl_grade <= 2:
            report += """For mild osteoarthritis, advanced treatments are usually not necessary initially.
However, if conservative management fails after 3-6 months, discuss these options 
with your physician:

• Intra-articular injections:
  - Corticosteroid injections for acute pain relief
  - Hyaluronic acid (viscosupplementation) series
  
• Regenerative medicine:
  - PRP (Platelet-Rich Plasma) therapy
  - Stem cell therapy (still investigational)
  
• Surgical options (rarely needed at this stage):
  - Arthroscopic debridement (if mechanical symptoms)
  - Osteotomy (if significant malalignment)
  
• When to consider referral to orthopedic surgeon:
  - Persistent pain despite 6 months conservative treatment
  - Significant limitation in daily activities
  - Mechanical symptoms (locking, catching)
"""
        
        elif kl_grade == 3:
            report += """For moderate osteoarthritis, you may benefit from advanced treatments:

• Conservative advanced options:
  - Intra-articular corticosteroid injections: 3-4 times per year
  - Hyaluronic acid injection series: Every 6 months
  - PRP therapy: Emerging option, discuss with physician
  
• Surgical considerations:
  - Arthroscopy: Limited benefit, mainly for mechanical issues
  - High tibial osteotomy: If younger with isolated medial disease
  - Partial knee replacement: For isolated compartment arthritis
  - Total knee replacement: Consider if conservative treatment fails
  
• Timing for orthopedic referral:
  - Persistent pain limiting daily activities despite treatment
  - Significant functional impairment
  - Progressive symptoms over 3-6 months
  - Considering surgical evaluation
  
• Questions to discuss with orthopedic surgeon:
  - Am I a candidate for surgery?
  - What are the risks vs. benefits?
  - What is the expected outcome?
  - What is the recovery timeline?
"""
        
        else:  # Grade 4
            report += """For severe osteoarthritis, you should strongly consider orthopedic consultation:

• Conservative management (continue while considering surgery):
  - Injections for temporary relief
  - Assistive devices (cane, walker)
  - Pain medications as prescribed
  - Physical therapy for strength and function
  
• Surgical options - STRONGLY RECOMMENDED to discuss:
  - Total Knee Arthroplasty (Total Knee Replacement):
    * Most definitive treatment for severe OA
    * High success rate (>90% satisfaction)
    * Recovery: 3-6 months for full recovery
    * Longevity: 15-20+ years typically
  
  - Partial Knee Replacement:
    * If only one compartment severely affected
    * Faster recovery than total replacement
    * May convert to total replacement later if needed
  
• Pre-surgical optimization:
  - Optimize weight (if overweight)
  - Control diabetes and other medical conditions
  - Strengthen muscles before surgery
  - Stop smoking (improves healing)
  - Dental clearance (prevent infection risk)
  
• URGENT orthopedic referral needed
  
• Red flags requiring immediate surgical consultation:
  - Severe pain not controlled with medications
  - Inability to perform basic daily activities
  - Progressive deformity
  - Instability or frequent falls
  
• Expected surgical outcomes:
  - Significant pain relief (80-90% reduction)
  - Improved function and quality of life
  - Return to low-impact activities
  - Ability to resume most daily activities
"""
        
        report += "\n\n9. FOLLOW-UP AND MONITORING\n─────────────────────────────────────────────────────────────\n"
        
        if kl_grade <= 1:
            report += """• Primary care follow-up: 6-12 months (or sooner if symptoms change)
• X-ray imaging: Only if significant symptom progression
• Physical therapy: Initial evaluation, then as needed
• Self-monitoring: Track pain levels, function, activities
"""
        elif kl_grade == 2:
            report += """• Primary care follow-up: 3-6 months (more frequent if significant symptoms)
• Orthopedic consultation: Consider if no improvement after 3 months
• Physical therapy: 6-12 sessions initially, periodic reassessment
• X-ray imaging: Annually or if significant changes in symptoms
• Self-monitoring: Pain diary, functional assessments
"""
        else:  # Grades 3-4
            report += """• Orthopedic consultation: Schedule soon (within 1-3 months)
• Primary care follow-up: Every 3 months or as symptoms dictate
• Physical therapy: Ongoing as tolerated and beneficial
• X-ray imaging: Annually or before surgical consultation
• Pain management: Regular assessment and adjustment
• Functional assessment: Monitor ability to perform daily activities
"""
        
        report += """
• Warning signs - contact healthcare provider if you experience:
  - Sudden increase in pain or swelling
  - New inability to bear weight
  - Fever with knee pain or redness (possible infection)
  - Significant decrease in function
  - Locking or instability of knee
  - Pain not controlled with recommended medications
  
• Expected outcomes with adherence to this plan:
"""
        
        if kl_grade <= 1:
            report += """  - Slowed or halted disease progression
  - Maintained or improved function
  - Minimal impact on daily activities
  - Good long-term prognosis with lifestyle modifications
"""
        elif kl_grade == 2:
            report += """  - Reduced pain and improved function within 4-12 weeks
  - Slowed disease progression
  - Improved ability to perform daily activities
  - May avoid or delay need for surgery with consistent adherence
"""
        else:
            report += """  - Improved pain control and function
  - Maintained mobility and independence
  - Informed decision about surgical options
  - Preparation for potential surgery if needed
"""
        
        report += "\n\n10. RESOURCES AND SUPPORT\n─────────────────────────────────────────────────────────────\n"
        
        report += """• Educational resources:
  - Arthritis Foundation (www.arthritis.org)
  - American Academy of Orthop aedic Surgeons (www.aaos.org)
  - Local arthritis support groups
  
• Specialist referrals:
  - Physical therapist (PT): For exercise program and manual therapy
  - Occupational therapist (OT): For adaptive strategies and equipment
  - Dietitian/nutritionist: For weight management and anti-inflammatory diet
  - Pain management specialist: If pain not controlled with conservative measures
  - Orthopedic surgeon: For advanced treatment options and surgical evaluation
  
• Community resources:
  - Arthritis water exercise programs at local pools
  - Senior centers with appropriate exercise classes
  - Online support groups and forums
  - Patient education programs at local hospitals
  
• Mobile apps and tools:
  - Pain tracking apps
  - Exercise reminder apps
  - Food/calorie tracking for weight management
  - Physical therapy home exercise programs

═══════════════════════════════════════════════════════════

IMPORTANT NOTES:
- This care plan is personalized based on your current status
- Adherence to recommendations is crucial for success
- Regular communication with healthcare team is essential
- Adjust plan as needed based on response and symptoms
- Don't hesitate to reach out if questions or concerns arise

Generated: {age}-year-old {sex if sex else 'patient'} with KL Grade {kl_grade} knee osteoarthritis
BMI: {bmi} ({bmi_category}) | Activity Level: {activity_level.capitalize()}

═══════════════════════════════════════════════════════════
"""
        
        return report.strip()
    
    def generate_complete_report(self, kl_grade, height, weight, age, activity_level, sex=None):
        """
        Complete workflow: Generate prescriptive care plan
        
        Returns:
            str: Complete prescriptive care report
        """
        print(f"Generating prescriptive care plan for KL Grade {kl_grade}...")
        
        report = self.generate_report_with_gemini(
            kl_grade, height, weight, age, activity_level, sex
        )
        
        return report


# Main function for easy integration
def generate_prescriptive_report(kl_grade, height, weight, age, activity_level, sex=None):
    """
    Main function to generate prescriptive care report
    
    Args:
        kl_grade: Kellgren-Lawrence grade (0-4)
        height: Patient height in cm
        weight: Patient weight in kg
        age: Patient age in years
        activity_level: Activity level ('sedentary', 'low', 'moderate', 'high', 'very high')
        sex: Patient sex ('Male'/'Female')
    
    Returns:
        str: Complete prescriptive care report
    """
    generator = PrescriptiveReportGenerator()
    return generator.generate_complete_report(
        kl_grade, height, weight, age, activity_level, sex
    )


if __name__ == '__main__':
    # Test report generation
    print("Testing Prescriptive Report Generator")
    print("=" * 60)
    
    # Test with different scenarios
    report = generate_prescriptive_report(
        kl_grade=2,
        height=175,
        weight=85,
        age=55,
        activity_level='moderate',
        sex='Male'
    )
    
    print(report[:500] + "...")

