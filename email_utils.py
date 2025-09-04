import os, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Iterable
from config import CFG


def _smtp_login(server: smtplib.SMTP):
if CFG.SMTP_USER and CFG.SMTP_PASS:
server.starttls()
server.login(CFG.SMTP_USER, CFG.SMTP_PASS)




def send_email(subject: str, body: str):
if not (CFG.EMAIL_FROM and CFG.EMAIL_TO):
raise RuntimeError("EMAIL_FROM/EMAIL_TO puuduvad")
msg = MIMEText(body, "plain", "utf-8")
msg["Subject"] = subject
msg["From"] = CFG.EMAIL_FROM
msg["To"] = CFG.EMAIL_TO
with smtplib.SMTP(CFG.SMTP_HOST, CFG.SMTP_PORT) as s:
_smtp_login(s)
s.sendmail(CFG.EMAIL_FROM, [CFG.EMAIL_TO], msg.as_string())




def send_email_html(subject: str, html: str, text: str | None = None, attachments: Iterable[str] | None = None):
if not (CFG.EMAIL_FROM and CFG.EMAIL_TO):
raise RuntimeError("EMAIL_FROM/EMAIL_TO puuduvad")
root = MIMEMultipart()
root["Subject"] = subject
root["From"] = CFG.EMAIL_FROM
root["To"] = CFG.EMAIL_TO


alt = MIMEMultipart('alternative')
root.attach(alt)
if text:
alt.attach(MIMEText(text, 'plain', 'utf-8'))
alt.attach(MIMEText(html, 'html', 'utf-8'))


for path in (attachments or []):
try:
with open(path, 'rb') as f:
part = MIMEBase('application', 'octet-stream')
part.set_payload(f.read())
encoders.encode_base64(part)
part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(path)}"')
root.attach(part)
except Exception:
continue


with smtplib.SMTP(CFG.SMTP_HOST, CFG.SMTP_PORT) as s:
_smtp_login(s)
s.sendmail(CFG.EMAIL_FROM, [CFG.EMAIL_TO], root.as_string())
