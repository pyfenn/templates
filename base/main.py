from fenn import FENN
from fenn.notification import Notifier
from fenn.notification.services import Discord, Telegram

app = FENN()
notifier = Notifier()
notifier.add_services([Discord, Telegram])

@app.entrypoint
def main(args):

    # 'args' contains your fenn.yaml configurations
    message = f"Training with learning rate: {args['training']['lr']}"

    notifier.notify(message)
    print(message)

    # Your logic here...

if __name__ == "__main__":
    app.run()