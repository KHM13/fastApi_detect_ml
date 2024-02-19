from datetime import datetime

today = datetime.today().strftime("%Y-%m-%d")


print(today)

test = "ddddddffffff"

print(test % f"d")