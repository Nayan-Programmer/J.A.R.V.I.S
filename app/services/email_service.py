"""
EMAIL SERVICE
=============
Sends emails via Gmail SMTP using an App Password.
Set in .env:
  GMAIL_ADDRESS      - your Gmail address
  GMAIL_APP_PASSWORD - a Gmail App Password (not your normal password)
"""

import logging
import os
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

logger = logging.getLogger("J.A.R.V.I.S")


def send_email(
    subject: str,
    body: str,
    to_address: Optional[str] = None,
    attachment_path=None,
) -> dict:
    gmail_address  = os.getenv("GMAIL_ADDRESS", "").strip()
    gmail_password = os.getenv("GMAIL_APP_PASSWORD", "").strip()

    if not gmail_address or not gmail_password:
        return {
            "success": False,
            "message": (
                "Gmail credentials are not set. "
                "Add GMAIL_ADDRESS and GMAIL_APP_PASSWORD to your .env file. "
                "Use a Gmail App Password (not your normal password)."
            ),
        }

    recipient = (to_address or gmail_address).strip()
    msg = MIMEMultipart()
    msg["From"]    = gmail_address
    msg["To"]      = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    if attachment_path and Path(attachment_path).exists():
        fname = Path(attachment_path).name
        with open(attachment_path, "rb") as f:
            part = MIMEApplication(f.read(), Name=fname)
        part["Content-Disposition"] = 'attachment; filename="' + fname + '"'
        msg.attach(part)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.login(gmail_address, gmail_password)
            server.sendmail(gmail_address, recipient, msg.as_string())
        logger.info("[EMAIL] Sent to %s | %s", recipient, subject)
        return {"success": True, "message": "Email sent to " + recipient + " successfully."}
    except smtplib.SMTPAuthenticationError:
        err = (
            "Gmail authentication failed. Make sure you are using a Gmail App Password. "
            "Go to myaccount.google.com/apppasswords to generate one."
        )
        logger.error("[EMAIL] Auth error")
        return {"success": False, "message": err}
    except Exception as exc:
        logger.error("[EMAIL] Failed: %s", exc)
        return {"success": False, "message": "Failed to send email: " + str(exc)}
