#coding:utf-8

# paddleocr --image_dir ./imgs_words/ch/word_1.jpg --det false


from paddleocr import PaddleOCR, draw_ocr
# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
cls_model_dir='paddleModels/whl/cls/ch_ppocr_mobile_v2.0_cls_infer'
rec_model_dir='paddleModels/whl/rec/ch/ch_PP-OCRv4_rec_infer'
ocr = PaddleOCR(use_angle_cls=True, lang="ch", det=False,cls_model_dir=cls_model_dir,rec_model_dir=rec_model_dir)  # need to run only once to download and load model into memory
img_path = '22.png'
result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    print(res)
    for line in res:

        print('-'*10)
        print(line)
        print(line[1])
        if '·' in line[1][0]:
            print('2222')
            print(line[1][0].replace('·',''))
print('000000')
res = result[0][0]

license_name, conf = result[0][0][1]
if '·' in license_name:
    license_name = license_name.replace('·', '')
print(license_name,conf)