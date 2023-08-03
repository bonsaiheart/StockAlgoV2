import datetime
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import tweepy
import PrivateData.twitter_info
from Task_Queue import celery_client

last_tweet_time = None
min_tweet_interval = datetime.timedelta(minutes=20)  # Minimum interval between tweets (5 minutes)


def send_tweet_w_countdown_followup(ticker, current_price, upordown, message, countdownseconds,modelname):
    global last_tweet_time

    directory = "last_tweet_timestamps"  # Directory for storing timestamp files
    os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
    timestamp_file_path = os.path.join(directory,f"last_tweet_timestamp_{modelname}.txt")  # File path inside the directory    global last_tweet_time
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
        pass

    if last_tweet_time is None or (current_time - last_tweet_time) >= min_tweet_interval:
        # Send the tweet

        last_tweet_time = current_time
        with open(timestamp_file_path, "w") as file:
            file.write(last_tweet_time.isoformat())
        try:
            response = client.create_tweet(text=message)

            tweet_id = response.data["id"]

            print("Tweet ID:", tweet_id)
            # wait_60_minutes_and_send_tweet("test3")

            celery_client.send_to_celery_1_hour(ticker, current_price, tweet_id, upordown, countdownseconds)
        except Exception as e:
            print(e)
            print(last_tweet_time, "last tweet time")


def email_me_string(strat, callorput, ticker):
    # Email configuration
    message = strat
    smtp_host = "bonsaiheart.com"
    smtp_port = 587
    smtp_user = "bot@bonsaiheart.com"
    smtp_password = "P3ruv!4nT0rch"
    from_email = "bot@bonsaiheart.com"
    to_email = "bot@bonsaiheart.com"
    subject = f"{str(callorput)} Ticker:{str(ticker)}"

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


# print(send_tweet_w_countdown_followup('tsla', 100, 'up', "A3test_response", 12,"testmodel4"))
