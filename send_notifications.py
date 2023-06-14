import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import tweepy
import PrivateData.twitter_info
from Task_Queue import celery_client


from celery import Celery


# def reply_tweet_1_hour_later(message,tweet_id):
#
#
#     bearer_token = PrivateData.twitter_info.bearer_token
#     consumer_key = PrivateData.twitter_info.consumer_key
#     consumer_secret = PrivateData.twitter_info.consumer_secret
#     access_token = PrivateData.twitter_info.access_token
#     access_token_secret = PrivateData.twitter_info.access_token_secret
#     api_key = PrivateData.twitter_info.api_key
#     api_key_secret = PrivateData.twitter_info.api_key_secret
#
#     client = tweepy.Client(bearer_token=bearer_token, consumer_key=consumer_key, consumer_secret=consumer_secret,
#                            access_token=access_token, access_token_secret=access_token_secret)
#
#
#     response = client.create_tweet(text=message,in_reply_to_tweet_id=tweet_id)
#
#     tweet_id = response.data['id']
#
#     print('Tweet ID:', tweet_id)
#     # wait_60_minutes_and_send_tweet("test3")




def send_tweet(ticker,current_price,upordown,message):


    bearer_token = PrivateData.twitter_info.bearer_token
    consumer_key = PrivateData.twitter_info.consumer_key
    consumer_secret = PrivateData.twitter_info.consumer_secret
    access_token = PrivateData.twitter_info.access_token
    access_token_secret = PrivateData.twitter_info.access_token_secret
    api_key = PrivateData.twitter_info.api_key
    api_key_secret = PrivateData.twitter_info.api_key_secret

    client = tweepy.Client(bearer_token=bearer_token, consumer_key=consumer_key, consumer_secret=consumer_secret,
                           access_token=access_token, access_token_secret=access_token_secret)


    response = client.create_tweet(text=message)

    tweet_id = response.data['id']

    print('Tweet ID:', tweet_id)
    # wait_60_minutes_and_send_tweet("test3")

    celery_client.send_to_celery(ticker, current_price, tweet_id, upordown)

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
# send_tweet("spy","3","up","test")
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
