import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional, List


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v


def send_email_html(
    subject: str,
    html: str,
    text: Optional[str] = None,
    attachments: Optional[List[str]] = None,
) -> None:
    """
    Send HTML email with optional plain-text alternative and attachments.
    Required env/secrets:
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_TO
    Optional:
      EMAIL_FROM (defaults to SMTP_USER)
    """
    smtp_host = _env("SMTP_HOST")
    smtp_port_raw = _env("SMTP_PORT", "587")
    smtp_user = _env("SMTP_USER")
    smtp_pass = _env("SMTP_PASS")
    email_from = _env("EMAIL_FROM", smtp_user)
    email_to_raw = _env("EMAIL_TO")

    if not smtp_host or not smtp_user or not smtp_pass or not email_to_raw:
        raise RuntimeError(
            "SMTP credentials or recipient missing. "
            "Set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_TO (and optionally EMAIL_FROM)."
        )

    try:
        smtp_port = int(smtp_port_raw or "587")
    except ValueError:
        smtp_port = 587

    # Parse recipients (comma or semicolon separated)
    recipients = [r.strip() for r in email_to_raw.replace(";", ",").split(",") if r.strip()]
    if not recipients:
        raise RuntimeError("EMAIL_TO is empty after parsing recipients.")

    # Build message
    msg = MIMEMultipart("alternative")
    msg["From"] = email_from
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject

    if text:
        msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))

    # Attach files
    if attachments:
        for file_path in attachments:
            if not file_path or not os.path.isfile(file_path):
                continue
            with open(file_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(file_path)}"')
            msg.attach(part)

    # Send (TLS for 587, SSL for 465)
    if smtp_port == 465:
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(email_from, recipients, msg.as_string())
    else:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            try:
                server.starttls()
                server.ehlo()
            except smtplib.SMTPException:
                # some servers already require TLS/SSL or don't support STARTTLS; continue without
                pass
            server.login(smtp_user, smtp_pass)
            server.sendmail(email_from, recipients, msg.as_string())
