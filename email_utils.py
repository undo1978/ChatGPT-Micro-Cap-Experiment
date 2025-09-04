import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional, List

from config import CFG


def send_email_html(
    subject: str,
    html: str,
    text: Optional[str] = None,
    attachments: Optional[List[str]] = None,
) -> None:
    """
    Send HTML email with optional plain-text alternative and attachments.
    """

    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", CFG.get("SMTP_USER"))
    smtp_pass = os.getenv("SMTP_PASS", CFG.get("SMTP_PASS"))
    email_from = os.getenv("EMAIL_FROM", smtp_user)
    email_to = os.getenv("EMAIL_TO", "")

    if not smtp_host or not smtp_user or not smtp_pass or not email_to:
        raise RuntimeError("SMTP credentials or recipient missing in environment/secrets")

    # build message
    msg = MIMEMultipart("alternative")
    msg["From"] = email_from
    msg["To"] = email_to
    msg["Subject"] = subject

    if text:
        msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))

    # add attachments if any
    if attachments:
        for file_path in attachments:
            if not os.path.isfile(file_path):
                continue
            with open(file_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(file_path)}"')
            msg.attach(part)

    # send
    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(email_from, [email_to], msg.as_string())
