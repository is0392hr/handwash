import qrcode
import cv2

qr = qrcode.QRCode()
qr.add_data('test text')
qr.make()
img = qr.make_image()
img.save('QR_img/test.png')
cv2.imshow('SCORE in QR',cv2.imread('QR_img/test.png'))
cv2.waitKey(0)
