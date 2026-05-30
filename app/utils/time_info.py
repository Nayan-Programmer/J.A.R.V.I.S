from datetime import datetime
from zoneinfo import ZoneInfo

def get_time_information():
    now = datetime.now(ZoneInfo("Asia/Kolkata"))

    return (
        f"Current Real-time Information:\n"
        f"Day: {now.strftime('%A')}\n"
        f"Date: {now.strftime('%d')}\n"
        f"Month: {now.strftime('%B')}\n"
        f"Year: {now.strftime('%Y')}\n"
        f"Time: {now.strftime('%H')} hours, {now.strftime('%M')} minutes, {now.strftime('%S')} seconds\n"
    )
