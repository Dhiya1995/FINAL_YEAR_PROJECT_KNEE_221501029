from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, 
                                PageBreak, Table, TableStyle, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage
import os
from datetime import datetime
from config import OUTPUT_FOLDER

def register_unicode_fonts():
    fonts_to_try = [
        ('C:/Windows/Fonts/Nirmala.ttc', 'Nirmala'),
        ('C:/Windows/Fonts/NirmalaUI.ttf', 'Nirmala'),
        ('C:/Windows/Fonts/nirmala.ttf', 'Nirmala'),
        
        ('C:/Windows/Fonts/mangal.ttf', 'Mangal'),
        ('C:/Windows/Fonts/Mangal.ttf', 'Mangal'),
        
        ('C:/Windows/Fonts/arialuni.ttf', 'ArialUnicode'),
        ('C:/Windows/Fonts/ARIALUNI.TTF', 'ArialUnicode'),
        
        ('C:/Windows/Fonts/NotoSans-Regular.ttf', 'NotoSans'),
        
        ('/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf', 'NotoSans'),
        ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'DejaVuSans'),
        
        ('C:/Windows/Fonts/Arial.ttf', 'Arial'),
    ]
    
    registered_font = None
    
    for font_path, font_name in fonts_to_try:
        try:
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                print(f"✅ Registered font: {font_name} - Supports English, Tamil, Hindi, French")
                registered_font = font_name
                break
        except Exception as e:
            print(f"⚠️ Could not register {font_name}: {e}")
            continue
    
    if not registered_font:
        print("⚠️ No Unicode font found, using Helvetica (English/French only)")
        registered_font = 'Helvetica'
    
    return registered_font

UNICODE_FONT = register_unicode_fonts()


def parse_llm_report_sections(report_text):
    sections = {}
    current_section = None
    current_content = []
    
    lines = report_text.split('\n')
    
    for line in lines:
        line_stripped = line.strip()
        
        if not line_stripped:
            continue
        
        is_header = False
        section_name = None
        
        if line_stripped.startswith('**') and ':' in line_stripped:
            clean_line = line_stripped.replace('**', '').strip()
            if clean_line.endswith(':'):
                potential_section = clean_line[:-1].strip()
                if '. ' in potential_section[:5]:
                    parts = potential_section.split('.', 1)
                    if parts[0].strip().isdigit():
                        potential_section = parts[1].strip()
                
                is_header = True
                section_name = potential_section.upper()
        
        elif line_stripped.endswith(':') and len(line_stripped) < 80:
            potential_section = line_stripped[:-1].strip()
            if potential_section.isupper() or len(potential_section.split()) <= 6:
                is_header = True
                section_name = potential_section.upper()
        
        if not is_header:
            parts = line_stripped.split('.', 1)
            if len(parts) == 2 and parts[0].strip().isdigit():
                potential_section = parts[1].strip()
                if len(potential_section) < 80 and (potential_section.isupper() or potential_section.endswith(':')):
                    is_header = True
                    section_name = potential_section.rstrip(':').upper()
        
        if is_header and section_name:
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content)
            
            current_section = section_name
            current_content = []
        else:
            if current_section:
                current_content.append(line_stripped)
            elif line_stripped and not line_stripped.startswith('#'):
                if 'INITIAL_CONTENT' not in sections:
                    sections['INITIAL_CONTENT'] = []
                if isinstance(sections['INITIAL_CONTENT'], list):
                    sections['INITIAL_CONTENT'].append(line_stripped)
    
    if current_section and current_content:
        sections[current_section] = '\n'.join(current_content)
    
    if 'INITIAL_CONTENT' in sections and isinstance(sections['INITIAL_CONTENT'], list):
        sections['FINDINGS'] = '\n'.join(sections['INITIAL_CONTENT'])
        del sections['INITIAL_CONTENT']
    
    return sections


def create_radiology_pdf_llm(report_text, patient_info, kl_grade, confidence, 
                             gradcam_path=None, output_path=None, 
                             language='en', prediction_id=None):
    
    print(f"📄 PDF Generator received report text ({len(report_text)} chars)")
    print(f"📄 First 200 chars: {report_text[:200]}...")
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_FOLDER, f"radiology_report_{timestamp}.pdf")
    
    doc = SimpleDocTemplate(
        output_path, 
        pagesize=letter,
        leftMargin=0.75*inch, 
        rightMargin=0.75*inch,
        topMargin=1*inch, 
        bottomMargin=0.75*inch
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Title'],
        fontSize=18,
        textColor=colors.HexColor('#1a5490'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName=UNICODE_FONT,
        leading=22
    )
    
    section_heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#1a5490'),
        spaceAfter=8,
        spaceBefore=14,
        fontName=UNICODE_FONT,
        leading=14,
        underline=True
    )
    
    normal_style = ParagraphStyle(
        'ReportNormal',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        leading=14,
        fontName=UNICODE_FONT
    )
    
    story.append(Paragraph("KNEE RADIOLOGICAL REPORT", title_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1a5490')))
    story.append(Spacer(1, 0.15*inch))
    
    report_date = datetime.now().strftime('%B %d, %Y')
    report_id = f"KOA-{datetime.now().strftime('%Y%m%d')}-{prediction_id if prediction_id else '0000'}"
    
    kl_diagnoses = {
        0: "No Osteoarthritis",
        1: "Doubtful Osteoarthritis",
        2: "Mild Osteoarthritis",
        3: "Moderate Osteoarthritis",
        4: "Severe Osteoarthritis"
    }
    
    diagnosis_text = f"Kellgren-Lawrence Grade {kl_grade}: {kl_diagnoses.get(kl_grade, 'OA')}"
    
    metadata_data = [
        ['Report Date:', report_date, 'Report ID:', report_id],
        ['Patient ID:', patient_info.get('Name', 'N/A'), 'Age/Sex:', 
         f"{patient_info.get('Age', 'N/A')}/{patient_info.get('Sex', 'N/A')}"],
        ['DIAGNOSIS:', diagnosis_text, '', '']
    ]
    
    metadata_table = Table(metadata_data, colWidths=[1.5*inch, 2.25*inch, 1.25*inch, 2*inch])
    metadata_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e6f2ff')),
        ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#e6f2ff')),
        ('FONTNAME', (0, 0), (0, -1), UNICODE_FONT),
        ('FONTNAME', (2, 0), (2, -1), UNICODE_FONT),
        ('SPAN', (0, 2), (1, 2)),
        ('SPAN', (2, 2), (3, 2)),
        ('BACKGROUND', (0, 2), (1, 2), colors.HexColor('#d4e6f7')),
        ('FONTNAME', (0, 2), (1, 2), UNICODE_FONT),
        ('TEXTCOLOR', (0, 2), (1, 2), colors.HexColor('#1a5490')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    story.append(metadata_table)
    story.append(Spacer(1, 0.2*inch))
    
    sections = parse_llm_report_sections(report_text)
    
    print(f"📄 Report text length: {len(report_text)} chars")
    print(f"🔍 PDF Parser found {len(sections)} sections")
    
    if len(sections) < 3:
        print("⚠️ Few/no sections parsed! Rendering entire report as paragraphs")
        
        paragraphs = report_text.split('\n\n')
        rendered_para_count = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            lines_in_para = para.split('\n')
            first_line = lines_in_para[0].strip() if lines_in_para else ""
            
            clean_first_line = first_line.replace('**', '').strip()
            
            is_potential_header = (
                (first_line.startswith('**') and ':' in first_line and len(lines_in_para) == 1) or
                (clean_first_line.isupper() and len(clean_first_line) < 80 and len(lines_in_para) == 1) or
                (clean_first_line.endswith(':') and len(clean_first_line) < 80 and len(lines_in_para) == 1)
            )
            
            if is_potential_header:
                header_text = clean_first_line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(header_text, section_heading_style))
                rendered_para_count += 1
            else:
                if '\n' in para:
                    for line in lines_in_para:
                        line = line.strip()
                        if not line:
                            continue
                        
                        line = line.replace('**', '')
                        
                        if line.startswith('* ') or line.startswith('*\t'):
                            line = '• ' + line[2:]
                        elif line.startswith('- '):
                            line = '• ' + line[2:]
                        
                        line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        
                        story.append(Paragraph(line, normal_style))
                        rendered_para_count += 1
                else:
                    clean_para = para.replace('**', '')
                    if clean_para.startswith('* ') or clean_para.startswith('*\t'):
                        clean_para = '• ' + clean_para[2:]
                    elif clean_para.startswith('- '):
                        clean_para = '• ' + clean_para[2:]
                    
                    clean_para = clean_para.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    story.append(Paragraph(clean_para, normal_style))
                    rendered_para_count += 1
            
            story.append(Spacer(1, 0.08*inch))
        
        print(f"✅ Rendered {rendered_para_count} paragraphs/lines to PDF")
    
    else:
        print(f"✅ Rendering {len(sections)} sections")
        
        section_order = [
            'DISCLAIMER',
            'CLINICAL HISTORY',
            'TECHNIQUE',
            'FINDINGS',
            'IMPRESSION',
            'RECOMMENDATIONS',
            'RISK PROFILE',
            'ESSENTIAL PRECAUTIONS',
            'PRECAUTIONS',
            'TREATMENT OPTIONS',
            'FOLLOW-UP AND MONITORING',
            'FOLLOW-UP'
        ]
        
        rendered_sections = set()
        for section_key in section_order:
            if section_key in sections:
                rendered_sections.add(section_key)
                
                header = section_key.title() if section_key != 'DISCLAIMER' else 'Disclaimer'
                story.append(Paragraph(f"{header}:", section_heading_style))
                
                content = sections[section_key]
                
                if section_key == 'FINDINGS' and gradcam_path and os.path.exists(gradcam_path):
                    try:
                        img = PILImage.open(gradcam_path)
                        max_img_width = 2.5 * inch
                        max_img_height = 2.5 * inch
                        aspect = img.width / img.height
                        
                        if aspect > 1:
                            img_width = max_img_width
                            img_height = max_img_width / aspect
                        else:
                            img_height = max_img_height
                            img_width = max_img_height * aspect
                        
                        gradcam_img = Image(gradcam_path, width=img_width, height=img_height)
                        findings_para = Paragraph(content, normal_style)
                        
                        findings_table = Table(
                            [[findings_para, gradcam_img]],
                            colWidths=[4*inch, 2.5*inch]
                        )
                        findings_table.setStyle(TableStyle([
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                            ('ALIGN', (1, 0), (1, 0), 'CENTER'),
                        ]))
                        
                        story.append(findings_table)
                    except Exception as e:
                        print(f"⚠️ Error loading image: {e}")
                        story.append(Paragraph(content, normal_style))
                else:
                    for line in content.split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        
                        line = line.replace('**', '')
                        
                        if line.startswith('* ') or line.startswith('*\t'):
                            line = '• ' + line[2:]
                        elif line.startswith('- '):
                            line = '• ' + line[2:]
                        
                        line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        
                        story.append(Paragraph(line, normal_style))
                
                story.append(Spacer(1, 0.1*inch))
        
        for section_key in sections.keys():
            if section_key not in rendered_sections:
                print(f"📝 Rendering extra section: {section_key}")
                story.append(Paragraph(f"{section_key.title()}:", section_heading_style))
                content = sections[section_key]
                for line in content.split('\n'):
                    line = line.strip()
                    if line:
                        story.append(Paragraph(line, normal_style))
                story.append(Spacer(1, 0.1*inch))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 0.1*inch))
    
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        alignment=TA_CENTER,
        textColor=colors.grey
    )
    
    footer_text = f"Generated by AI-Powered Knee Osteoarthritis Analysis System | {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
    story.append(Paragraph(footer_text, footer_style))
    
    doc.build(story)
    
    return output_path


def create_prescriptive_pdf_llm(report_text, patient_info, kl_grade, output_path=None):
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_FOLDER, f"prescriptive_care_{timestamp}.pdf")
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
        topMargin=1*inch,
        bottomMargin=0.75*inch
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CareTitle',
        parent=styles['Title'],
        fontSize=18,
        textColor=colors.HexColor('#2d8659'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName=UNICODE_FONT,
        leading=22
    )
    
    subtitle_style = ParagraphStyle(
        'CareSubtitle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2d8659'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName=UNICODE_FONT,
        leading=16
    )
    
    section_heading_style = ParagraphStyle(
        'CareSectionHeading',
        parent=styles['Heading2'],
        fontSize=11,
        textColor=colors.HexColor('#2d8659'),
        spaceAfter=8,
        spaceBefore=12,
        fontName=UNICODE_FONT,
        leading=13
    )
    
    normal_style = ParagraphStyle(
        'CareNormal',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_LEFT,
        spaceAfter=6,
        leading=13,
        fontName=UNICODE_FONT
    )
    
    story.append(Paragraph("PRESCRIPTIVE CARE PLAN", title_style))
    story.append(Paragraph("Knee Osteoarthritis Management", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2d8659')))
    story.append(Spacer(1, 0.15*inch))
    
    kl_diagnoses = {
        0: "No Osteoarthritis",
        1: "Doubtful Osteoarthritis",
        2: "Mild Osteoarthritis",
        3: "Moderate Osteoarthritis",
        4: "Severe Osteoarthritis"
    }
    
    patient_data = [
        ['Patient ID:', patient_info.get('Name', 'N/A')],
        ['Age:', patient_info.get('Age', 'N/A')],
        ['Sex:', patient_info.get('Sex', 'N/A')],
        ['Height:', patient_info.get('Height', 'N/A')],
        ['Weight:', patient_info.get('Weight', 'N/A')],
        ['Activity Level:', patient_info.get('Activity Level', 'N/A')],
        ['Diagnosis:', f"KL Grade {kl_grade} - {kl_diagnoses.get(kl_grade, 'OA')}"]
    ]
    
    patient_table = Table(patient_data, colWidths=[2*inch, 4.5*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f5e9')),
        ('FONTNAME', (0, 0), (0, -1), UNICODE_FONT),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    story.append(patient_table)
    story.append(Spacer(1, 0.2*inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 0.15*inch))
    
    lines = report_text.split('\n')
    
    for line in lines:
        line_stripped = line.strip()
        
        if not line_stripped or line_stripped.startswith('═'):
            continue
        
        clean_line = line_stripped.replace('**', '').strip()
        
        is_header = False
        if line_stripped.startswith('**') and ':' in line_stripped:
            is_header = True
        elif clean_line.endswith(':') and len(clean_line) < 80:
            is_header = True
        elif clean_line.isupper() and len(clean_line) < 80:
            is_header = True
        elif clean_line.split('.')[0].strip().isdigit() and len(clean_line) < 80 and ':' in clean_line:
            is_header = True
        
        if is_header:
            header_text = clean_line
            if '. ' in header_text[:5]:
                parts = header_text.split('.', 1)
                if parts[0].strip().isdigit():
                    header_text = parts[1].strip()
            
            header_text = header_text.rstrip(':')
            
            header_text = header_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            story.append(Spacer(1, 0.05*inch))
            story.append(Paragraph(header_text + ':', section_heading_style))
        else:
            clean_content = line_stripped.replace('**', '')
            
            if clean_content.startswith('* ') or clean_content.startswith('*\t'):
                clean_content = '• ' + clean_content[2:]
            elif clean_content.startswith('- '):
                clean_content = '• ' + clean_content[2:]
            
            clean_content = clean_content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            story.append(Paragraph(clean_content, normal_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 0.1*inch))
    
    footer_style = ParagraphStyle(
        'CareFooter',
        parent=styles['Normal'],
        fontSize=8,
        alignment=TA_CENTER,
        textColor=colors.grey
    )
    
    footer_text = f"Generated by AI-Powered Prescriptive Care System | {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
    story.append(Paragraph(footer_text, footer_style))
    
    doc.build(story)
    
    return output_path


def create_pdf_report(report_text, patient_info, kl_grade, confidence, gradcam_path=None,
                     output_path=None, language='en', prediction_id=None):
    return create_radiology_pdf_llm(
        report_text, patient_info, kl_grade, confidence,
        gradcam_path, output_path, language, prediction_id
    )


def create_prescriptive_pdf(report_text, patient_info, output_path=None):
    kl_grade = patient_info.get('KL_Grade', 0)
    return create_prescriptive_pdf_llm(report_text, patient_info, kl_grade, output_path)


if __name__ == '__main__':
    print("Testing LLM-First PDF Generation")
    
    sample_radiology_report = """DISCLAIMER:
This is an AI-generated report requiring confirmation by a board-certified radiologist.

CLINICAL HISTORY:
65-year-old Male patient presenting with knee pain for 6 months.

TECHNIQUE:
Standard anteroposterior and lateral radiographic views of the knee were obtained.
AI Classification Confidence: 87.5%
Kellgren-Lawrence Grading System Applied

FINDINGS:
Definite joint space narrowing is present in the medial compartment.
Small osteophytes are identified at the joint margins.
Mild subchondral sclerosis is noted.
Joint alignment is maintained.

IMPRESSION:
Mild knee osteoarthritis (Kellgren-Lawrence Grade 2).

RECOMMENDATIONS:
Conservative management including physical therapy and NSAIDs recommended.
Orthopedic follow-up as needed."""
    
    patient_info = {
        'Name': 'Test Patient',
        'Age': '65',
        'Sex': 'Male'
    }
    
    pdf_path = create_radiology_pdf_llm(
        sample_radiology_report, patient_info, 2, 0.875,
        gradcam_path=None, prediction_id=1
    )
    
    print(f"Test PDF created: {pdf_path}")
