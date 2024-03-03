# Import required library 
from keras.models import load_model

import matplotlib.pyplot as plt

from io import BytesIO

# Import smtplib for the actual sending function.
import smtplib

# Here are the email package modules we'll need.
from email.message import EmailMessage

piCamera = False
if picamera == True:
    from picamera import PiCamera
    import time
    from picamera.array import PiRGBArray

import argparse
import script.utils

def parse_args():
    desc = "Plant disease Classification"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=str, default='model/model__Alexnet.h5', help='Where Is Model File?')
    parser.add_argument('--img', type=str, default='data/1.jpg', help='What Is Images Path?')

    return parser.parse_args()

def convertToJpeg(im):
    with BytesIO() as f:
        im.save(f, format='JPEG')
        return f.getvalue()

def main():
    args = parse_args()
    if args is None:
        exit()
    try:
        while True:
            
            # Load Model
            model = load_model(args.model)
            # Convert Image To Numpy Array
            if picamera == True:
                camera = PiCamera()
                rawImage = PiRGBArray(camera)
                time.sleep(0.1)
                camera.capture(rawImage, format = "rgb")
                image = rawImage.array
            else:
                image = script.utils.load_image(args.img)
            # Predict Image Based On Model
            label = model.predict(image)
            # Print Result
            print("Predicted Class (0 - No disease in leaf , xx- Disease in leaf): ", round(label[0][0], 2))
            if round(label[0][0], 2) == 0:
                # displaying the image 
                plt.imshow(image)
                plt.title('No disease in leaf',  
                                     fontweight ="bold")
                plt.show()
                print("No disease in leaf")
            else:
                # displaying the image 
                plt.imshow(image)
                plt.title('Disease in leaf',  
                                     fontweight ="bold")
                plt.show()
                print("Disease in leaf")

                # Create the container email message.
                msg = EmailMessage()
                msg['Subject'] = 'Disease in leaf'
                msg['From'] = email_from
                msg['To'] = email_to
                #msg.preamble = 'You will not see this in a MIME-aware mail reader.\n'
                img_data = convertToJpeg(image)
                msg.add_attachment(img_data, maintype='image',
                                 subtype='jpeg')

                # Send the email via our own SMTP server.
                with smtplib.SMTP('localhost') as s:
                    s.send_message(msg)
                
            if picamera == False:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
