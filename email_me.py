import smtplib

# from email.mime.text import MIMEText
# from email.utils import COMMASPACE
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


def email_me_string(order,statuscode,response):
    note, price, quantity, contract, stopcoefficient, sellcoefficient = order
    # Email configuration
    message = note
    smtp_host = "bonsaiheart.com"
    smtp_port = 587
    smtp_user = "bot@bonsaiheart.com"
    smtp_password = "P3ruv!4nT0rch"
    from_email = "bot@bonsaiheart.com"
    to_email = "bot@bonsaiheart.com"
    subject =f"{str(contract)} Price:{str(price)} {str(response)}"

    # Create the email message
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject

    # Set the message content
    body = MIMEText(message)
    msg.attach(body)

    # Send the email using SMTP
    server = smtplib.SMTP(smtp_host, smtp_port)
    server.starttls()
    server.login(smtp_user, smtp_password)
    server.sendmail(from_email, to_email, msg.as_string())
    server.quit()

    print("Email sent!")

def email_me(filepath):
    # Email configuration
    smtp_host = "bonsaiheart.com"
    smtp_port = 587
    smtp_user = "bot@bonsaiheart.com"
    smtp_password = "P3ruv!4nT0rch"
    from_email = "bot@bonsaiheart.com"
    to_email = "bot@bonsaiheart.com"
    subject = f"{filepath}"

    # Create the email message
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject

    # Attach the CSV file to the email message
    csv_file_path = filepath
    with open(csv_file_path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{csv_file_path.split("/")[-1]}"')
        msg.attach(part)

    # Send the email using SMTP
    server = smtplib.SMTP(smtp_host, smtp_port)
    server.starttls()
    server.login(smtp_user, smtp_password)
    server.sendmail(from_email, to_email, msg.as_string())
    server.quit()

    print("Email sent!")
