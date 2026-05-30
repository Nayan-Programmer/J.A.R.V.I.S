"""
JARVIS EMAIL SERVICE (STABLE VERSION)
=====================================
Uses Gmail SMTP over SSL (port 465)
Fixes network issues better than STARTTLS.

.env required:
  GMAIL_ADDRESS=yourgmail@gmail.com
  GMAIL_APP_PASSWORD=your16digitapppassword
"""

import os
import smtplib
import logging
import socket
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("J.A.R.V.I.S")


# ---------------------------
# Email validation (simple)
# ---------------------------
def is_valid_email(email: str) -> bool:
    return "@" in email and "." in email and " " not in email


# ---------------------------
# MAIN FUNCTION
# ---------------------------
def send_email(subject: str, body: str, to_address: str, attachment_path=None) -> dict:

    gmail_address = os.getenv("GMAIL_ADDRESS", "").strip()
    gmail_password = os.getenv("GMAIL_APP_PASSWORD", "").replace(" ", "").strip()

    # ---------------------------
    # Safety checks
    # ---------------------------
    if not gmail_address or not gmail_password:
        return {
            "success": False,
            "message": "Missing Gmail credentials in .env"
        }

    if not is_valid_email(to_address):
        return {
            "success": False,
            "message": "Invalid recipient email"
        }

    recipient = to_address.strip()

    # ---------------------------
    # Build email
    # ---------------------------
    msg = MIMEMultipart()
    msg["From"] = gmail_address
    msg["To"] = recipient
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    # ---------------------------
    # Attachment support
    # ---------------------------
    if attachment_path:
        path = Path(attachment_path)
        if path.exists():
            with open(path, "rb") as f:
                part = MIMEApplication(f.read(), Name=path.name)
            part["Content-Disposition"] = f'attachment; filename="{path.name}"'
            msg.attach(part)

    # ---------------------------
    # Network stability fix
    # ---------------------------
    socket.setdefaulttimeout(30)

    # ---------------------------
    # SEND EMAIL (SSL FIX)
    # ---------------------------
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
            server.login(gmail_address, gmail_password)
            server.sendmail(gmail_address, recipient, msg.as_string())

        logger.info(f"[EMAIL] Sent successfully to {recipient}")

        return {
            "success": True,
            "message": f"Email sent to {recipient}"
        }

    except smtplib.SMTPAuthenticationError:
        logger.error("[EMAIL] Authentication failed")
        return {
            "success": False,
            "message": "Gmail auth failed. Use App Password."
        }

    except Exception as e:
        logger.error(f"[EMAIL] Failed: {e}")
        return {
            "success": False,
            "message": str(e)
        }
