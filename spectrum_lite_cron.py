import time
import os
import datetime
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler

from spectrum_lite import spectrum_lite_run

cronStartedAt = datetime.datetime.now()
print(f"Spectrum Lite cron job started @ {cronStartedAt}")

load_dotenv()

schedule_hour = 23
schedule_minute = 50

if "SCHEDULE_HOUR" in os.environ:
    schedule_hour = os.getenv("SCHEDULE_HOUR")

if "SCHEDULE_MINUTE" in os.environ:
    schedule_minute = os.getenv("SCHEDULE_MINUTE")

print("schedule_hour:", schedule_hour)
print("schedule_minute:", schedule_minute)

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    # run every day @ schedule_hour and schedule_minute in America/Los_Angeles - PST timezone
    scheduler.add_job(spectrum_lite_run, 'cron',
                      hour=schedule_hour, minute=schedule_minute,
                      timezone='America/Los_Angeles')
    scheduler.start()
    print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))

    try:
        # This is here to simulate application activity (which keeps the main thread alive).
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        # Not strictly necessary if daemonic mode is enabled but should be done if possible
        scheduler.shutdown()
