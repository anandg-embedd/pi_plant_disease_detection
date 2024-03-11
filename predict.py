# Import required library 
from keras.models import load_model

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from io import BytesIO

# Import smtplib for the actual sending function.
import smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


# Here are the email package modules we'll need.
from email.message import EmailMessage

import time
import numpy as np
from skimage import transform
from PIL import Image as img_convert 

tensor_lite_en = True
picamera = 1
webcamera = 2
nocamera = 0
camerain = webcamera
if camerain == picamera:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
elif camerain == webcamera:
    import cv2
    webcam = cv2.VideoCapture(0)
    time.sleep(2)

import argparse
import script.utils
img_width, img_height = 224, 224
def parse_args():
    desc = "Plant disease Classification"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=str, default='model/model__Alexnet.h5', help='Where Is Model File?')
    parser.add_argument('--img', type=str, default='data/1.jpg', help='What Is Images Path?')

    return parser.parse_args()

def convertToJpeg(im):
    data=img_convert.fromarray(im)
    with BytesIO() as f:
        data.save(f, format='JPEG')
        return f.getvalue()


def img_pre_process(np_image):
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (img_width, img_height, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def send_attached_mail(to_address, subject, message, attachment):
    mail_user = "ammu201995@gmail.com"
    mail_password = "msaphdugxxbjyztw"

    send_from = mail_user
    send_to = [to_address, 'g.ganeshlex@gmail.com']
    subject = subject
    body = message
    file = attachment

    message = MIMEMultipart()
    message["From"] = mail_user
    message["To"] = to_address
    message["Subject"] = subject
    message["Bcc"] = to_address  # Recommend for mass emails

    # Add body to email

    message.attach(MIMEText(body, "plain"))

    filename = file  # In same directory as script

    # Open PDF file in binary node

    with open(filename, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename={filename}",
    )

    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()
    try:
        # Log in to server using secure context and send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(mail_user, mail_password)
            server.sendmail(send_from, send_to, text)
        return "Email has been Successfully send."
    except Exception as ex:
        return "Something went wrong..." + str(ex)


def main():
    args = parse_args()
    if args is None:
        exit()
    try:
        while True:
            

            if camerain == picamera:
                # Convert Image To Numpy Array
                camera = PiCamera()
		camera.resolution = (img_width, img_height)
                rawImage = PiRGBArray(camera)
                time.sleep(0.1)
                camera.capture(rawImage, format = "rgb")
                disp_img = rawImage.array
		camera.close()
                image = img_pre_process(disp_img)
                #camera = PiCamera()
                #camera.resolution = (img_width, img_height)
                #camera.start_preview()
                #time.sleep(5) # hang for preview for 5 seconds
                #camera.capture('data/1.jpg')
                #camera.stop_preview()
                #camera.close()
                #time.sleep(5)
                #print('load image')
                #image = script.utils.load_image('data/1.jpg')
                #disp_img=mpimg.imread('data/1.jpg')
            elif camerain == webcamera:
                try:
                    check, frame = webcam.read()
                    disp_img=frame;
                    image = img_pre_process(frame)
                except(KeyboardInterrupt):
                    print("Turning off camera.")
                    webcam.release()
                    print("Camera off.")
                    print("Program ended.")
                    cv2.destroyAllWindows()
                    break
            else:
                image = script.utils.load_image(args.img)
                disp_img=mpimg.imread(args.img)
            if tensor_lite_en == True:
                from tflite_runtime.interpreter import Interpreter

                # Load the TFLite model and allocate tensors.
                interpreter = Interpreter(model_path="model.tflite")
                interpreter.allocate_tensors()

                # Get input and output tensors.
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                # Test the model on random input data.
                # input_shape = input_details[0]['shape']
                input_data = image # np.array(np.random.random_sample(input_shape), dtype=np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)

                interpreter.invoke()

                # The function `get_tensor()` returns a copy of the tensor data.
                # Use `tensor()` in order to get a pointer to the tensor.
                label = interpreter.get_tensor(output_details[0]['index'])
                print(label)
            else:
                from keras.models import load_model
                # Load Model
                model = load_model(args.model)
                # Predict Image Based On Model
                label = model.predict(image)
            # Print Result
            print("Predicted Class (0 - No disease in leaf , xx- Disease in leaf): ", round(label[0][0], 2))
            if round(label[0][0], 2) == 0:
                # displaying the image 
                plt.imshow(disp_img)
                plt.title('No disease in leaf',  
                                     fontweight ="bold")
                plt.show()
                print("No disease in leaf")
                im = img_convert.fromarray(disp_img)
                im.save("data/res.jpeg")
                time.sleep(5)
                send_attached_mail('anandg.embedd@gmail.com', 'leaf status', 'Disease in leaf','data/res.jpeg')
            else:
                # displaying the image 
                plt.imshow(disp_img)
                plt.title('Disease in leaf',  
                                     fontweight ="bold")
                plt.show()
                print("Disease in leaf")
                im = img_convert.fromarray(disp_img)
                im.save("data/res.jpeg")
                time.sleep(5)
                # Create the container email message.
                send_attached_mail('anandg.embedd@gmail.com', 'leaf status', 'Disease in leaf','data/res.jpeg')
##                msg = EmailMessage()
##                msg['Subject'] = 'Disease in leaf'
##                msg['From'] = anandg.embedd@gmail.com
##                msg['To'] = email_to
##                #msg.preamble = 'You will not see this in a MIME-aware mail reader.\n'
##                img_data = convertToJpeg(image)
##                msg.add_attachment(img_data, maintype='image',
##                                 subtype='jpeg')
##
##                # Send the email via our own SMTP server.
##                with smtplib.SMTP('smtp.gmail.com',465) as s:
##                    s.send_message(msg)
                
            if camerain == nocamera:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
