import datetime
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import tweepy
import PrivateData
import PrivateData.email
from Task_Queue import celery_client
from UTILITIES.logger_config import logger
# min_tweet_interval = datetime.timedelta(minutes=60)  # Minimum interval between tweets (5 minutes)

def send_tweet_w_countdown_followup(ticker, current_price, upordown, message, countdownseconds, modelname):
    min_tweet_interval = datetime.timedelta(minutes=countdownseconds // 60)
    print("~~~Attempt Sending Tweet~~~")

    directory = "UTILITIES/Send_Notifications/last_tweet_timestamps"
    os.makedirs(directory, exist_ok=True)
    timestamp_file_path = os.path.join(directory, f"last_tweet_timestamp_{modelname}_{ticker}.txt")
    current_time = datetime.datetime.now()

    bearer_token = PrivateData.twitter_info.bearer_token
    consumer_key = PrivateData.twitter_info.consumer_key
    consumer_secret = PrivateData.twitter_info.consumer_secret
    access_token = PrivateData.twitter_info.access_token
    access_token_secret = PrivateData.twitter_info.access_token_secret
    client = tweepy.Client(
        bearer_token=bearer_token,
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
    )

    try:
        with open(timestamp_file_path, "r") as file:
            timestamp_str = file.read().strip()
            if timestamp_str:
                last_tweet_time = datetime.datetime.fromisoformat(timestamp_str)
    except FileNotFoundError:
        last_tweet_time = None

    if last_tweet_time is None or (current_time - last_tweet_time) >= min_tweet_interval:
        print("~~~Sending Tweet~~~")
        try:
            response = client.create_tweet(text=message)
            tweet_id = response.data["id"]
            print("Tweet ID:", tweet_id)

            with open(timestamp_file_path, "w") as file:
                file.write(current_time.isoformat())

            celery_client.send_to_celery_1_hour(ticker, current_price, tweet_id, upordown, countdownseconds)
        except Exception as e:
            print(f"Error while sending tweet: {e}")
            logger.error(f"An error occurred while trying to tweet for {ticker}: {e}", exc_info=True)
    else:
        print(last_tweet_time, "too close to last tweet time")


def email_me_string(model_name, callorput, ticker):

    message = model_name
    smtp_host = PrivateData.email.smtp_host
    smtp_port = PrivateData.email.smtp_port
    smtp_user = PrivateData.email.smtp_user
    smtp_password = PrivateData.email.smtp_password
    from_email = PrivateData.email.from_email
    to_email = PrivateData.email.to_email

    subject = f"{str(ticker)}* {message} "

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject

    body = MIMEText(f"{str(ticker)}* {message}" )
    msg.attach(body)

    server = smtplib.SMTP(smtp_host, smtp_port)
    server.starttls()
    server.login(smtp_user, smtp_password)
    server.sendmail(from_email, to_email, msg.as_string())
    server.quit()

    print("Email sent!")


# send_tweet("spy","3","up","test")

# email_me_string("Buy_1hr_Ptminfakerrrhah","C","SPY")
def email_me(filepath):
    # Email configuration
    smtp_host = PrivateData.email.smtp_host
    smtp_port = PrivateData.email.smtp_port
    smtp_user = PrivateData.email.smtp_user
    smtp_password = PrivateData.email.smtp_password
    from_email = PrivateData.email.from_email
    to_email = PrivateData.email.to_email
    subject = f"{filepath}"

    # create mesage
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject

    # Attach CSV content
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


# print(send_tweet_w_countdown_followup('tsla', 100, 'up', "A3test_response", 12,"testmodel4"))
