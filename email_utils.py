import os, smtplib
from email.mime.text import MIMEText


def send_email(subject: str, body: str):
host = os.getenv("SMTP_HOST")
port = int(os.getenv("SMTP_PORT", "587"))
user = os.getenv("SMTP_USER")
pwd = os.getenv("SMTP_PASS")
sender = os.getenv("EMAIL_FROM")
to = os.getenv("EMAIL_TO")


if not all([host, port, user, pwd, sender, to]):
raise RuntimeError("Email env muutujad puuduvad (SMTP_HOST/PORT/USER/PASS, EMAIL_FROM/TO)")


msg = MIMEText(body, "plain", "utf-8")
msg["Subject"] = subject
msg["From"] = sender
msg["To"] = to


with smtplib.SMTP(host, port) as s:
s.starttls()
s.login(user, pwd)
s.sendmail(sender, [to], msg.as_string())
