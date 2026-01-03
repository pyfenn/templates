from fenn import FENN
from fenn.notification import Notifier
from fenn.notification.services import Resend

app = FENN()

@app.entrypoint
def main(args):
    # 'args' contains your fenn.yaml configurations
    print(f"Training with learning rate: {args['train']['lr']}")
    
    # Setup email notifications
    notifier = Notifier()
    
    notifier.add_service(Resend)

    """# To Create Resend service with custom subject (subject configured in fenn.yaml)
    subject = args.get('notification', {}).get('subject', 'Notification from fenn')
    resend_service = Resend(subject=subject)
    notifier._services.append(resend_service)
    """
    
    # Send notification at start
    notifier.notify(f"Training started with lr={args['train']['lr']}, epochs={args['train']['epochs']}")
    
    # Your training logic here...
    for epoch in range(args['train']['epochs']):
        # Simulate training
        print(f"Epoch {epoch + 1}/{args['train']['epochs']}")
        
        # Send notification every 5 epochs
        if (epoch + 1) % 5 == 0:
            notifier.notify(f"Training progress: Epoch {epoch + 1}/{args['train']['epochs']} completed")
    
    # Send notification at completion
    notifier.notify(f"Training completed! Total epochs: {args['train']['epochs']}")

if __name__ == "__main__":
    app.run()