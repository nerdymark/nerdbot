#!/usr/bin/env python3
"""
Generate a PDF cheat sheet for Steam Controller joystick controls
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime

def create_cheatsheet():
    """Create the joystick control cheat sheet PDF"""

    # Create PDF document
    filename = "/home/mark/nerdbot-backend/NerdBot_Joystick_Controls.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter,
                          topMargin=0.5*inch, bottomMargin=0.5*inch,
                          leftMargin=0.75*inch, rightMargin=0.75*inch)

    # Container for the 'Flowable' objects
    elements = []

    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2E86AB'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#A23B72'),
        spaceAfter=12,
        spaceBefore=20,
        fontName='Helvetica-Bold'
    )

    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=13,
        textColor=colors.HexColor('#F18F01'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    # Title
    title = Paragraph("NerdBot Steam Controller<br/>Quick Reference Guide", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.2*inch))

    # Analog Controls Section
    elements.append(Paragraph("Analog Controls", heading_style))

    analog_data = [
        ['Control', 'Function', 'Details'],
        ['Left Analog Stick', 'Camera Pan/Tilt',
         '• X-axis (Left/Right): Pan camera\n• Y-axis (Up/Down): Tilt camera\n• Fine control for precise aiming'],
        ['Left Touchpad\n(with groove)', 'Robot Movement',
         '• Y-axis: Forward/Backward\n• X-axis: Strafe Left/Right\n• WASD-style movement\n• Supports simultaneous strafe + rotation'],
        ['Right Touchpad\n(circular)', 'Robot Rotation',
         '• X-axis: Rotation direction & speed\n• Distance from center = rotation speed\n• Left: Rotate counterclockwise\n• Right: Rotate clockwise'],
    ]

    analog_table = Table(analog_data, colWidths=[1.5*inch, 1.5*inch, 3.5*inch])
    analog_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')]),
    ]))

    elements.append(analog_table)
    elements.append(Spacer(1, 0.3*inch))

    # Button Controls Section
    elements.append(Paragraph("Button Controls", heading_style))

    button_data = [
        ['Button', 'Function', 'Notes'],
        ['A Button', 'Center Camera', 'Resets pan/tilt servos to default position'],
        ['B Button', 'Emergency Stop', 'Immediately stops all motor movement'],
        ['X Button', 'Cycle Robot Modes', 'Switches between operational modes'],
        ['Y Button', 'Random Meme Sound', '1 second cooldown between plays'],
        ['Right Shoulder', 'Welcome Message', 'Plays TTS welcome greeting'],
        ['Button 10', 'Toggle Headlights', '500ms cooldown'],
        ['Button 11', 'Toggle Laser', '500ms cooldown'],
    ]

    button_table = Table(button_data, colWidths=[1.5*inch, 2*inch, 3*inch])
    button_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#A23B72')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')]),
    ]))

    elements.append(button_table)
    elements.append(Spacer(1, 0.3*inch))

    # Tips Section
    elements.append(Paragraph("Pro Tips", heading_style))

    tips_data = [
        ['💡', 'Combined Movement', 'You can strafe, move forward, and rotate simultaneously for complex maneuvers'],
        ['💡', 'Deadzone', 'Small movements near center are ignored to prevent drift (15% deadzone)'],
        ['💡', 'Debouncing', 'Button cooldowns prevent accidental double-presses and audio overlap'],
        ['💡', 'Zero Latency', 'Joystick service directly controls motors/servos without HTTP overhead'],
        ['⚠️', 'Safety First', 'B button (Emergency Stop) immediately halts all motor activity'],
    ]

    tips_table = Table(tips_data, colWidths=[0.5*inch, 1.8*inch, 4.2*inch])
    tips_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTSIZE', (0, 0), (0, -1), 14),  # Emoji column
        ('FONTSIZE', (1, 0), (-1, -1), 9),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),  # Tip title column
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.HexColor('#FFF9E6')]),
    ]))

    elements.append(tips_table)
    elements.append(Spacer(1, 0.3*inch))

    # Technical Details
    elements.append(Paragraph("Technical Details", heading_style))

    tech_info = f"""
    <b>Service:</b> nerdbot-joystick.service<br/>
    <b>Update Rate:</b> 100Hz (10ms polling interval)<br/>
    <b>Control Type:</b> Mecanum wheel mathematics for omnidirectional movement<br/>
    <b>Architecture:</b> Direct hardware integration (bypasses Flask API)<br/>
    <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
    """

    tech_para = Paragraph(tech_info, styles['Normal'])
    elements.append(tech_para)

    # Footer
    elements.append(Spacer(1, 0.3*inch))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    footer = Paragraph("NerdBot Integrated Joystick Service | Raspberry Pi 5 + Steam Controller", footer_style)
    elements.append(footer)

    # Build PDF
    doc.build(elements)
    print(f"PDF created successfully: {filename}")
    return filename

if __name__ == "__main__":
    create_cheatsheet()
