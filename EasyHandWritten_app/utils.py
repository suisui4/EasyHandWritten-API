
# Judgeテーブルからカテゴリ名を返すやつ
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
import numpy as np
from EasyHandWritten_app.models import Judge

# AIジャッチ判定の名前Cdから変換
def get_category_names():
    categories = Judge.objects.all()
    category_dict = {category.id: category.judge_str for category in categories}
    return category_dict

# DB画像の画像をbase64に変換
def get_image(DBimage):
    if DBimage == None:
        return ""
    # 画像を Base64 に変換
    return base64.b64encode(DBimage).decode('utf-8')

#######----マルバツ判定
# PyTorchのCNNモデル
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 1はグレースケール画像
        self.conv2 = nn.Conv2d(16, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 7 * 7, 128)  # 画像サイズ28x28 -> 7x7にプール後のサイズ
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 256 * 7 * 7)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def get_judgemodel(judge_id):
    if judge_id=='1':
        return 'number.pth', ["0","1","2","3","4","5","6","7","8","9"]
    if judge_id=='2':
        return 'hiragana.pth', ["あ", "い", "う", "え", "お", "か", "き", "く", "け", "こ", "さ", "し", "す"]
    if judge_id=='3':
        return 'cicle_cross.pth', ["⚪︎", "×"]

# AI判定を行う関数
def IMAGE_TO_RESULT(image_data,judge_id):
    # 画像データをNumPy配列に変換 (バイナリデータ -> OpenCVで処理)
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # 画像の前処理（リサイズ、正規化など）
    img = cv2.resize(img, (28, 28))  # 28x28にリサイズ
    img = img.astype('float32') / 255  # 正規化

    # PyTorchのTensorに変換
    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)の形に変換
    model_judge, result_list = get_judgemodel(judge_id)
    # モデルのインスタンスを作成
    model = CNNModel(num_classes=len(result_list))  # クラス数は2（丸、バツ）
    model.load_state_dict(torch.load('studiedModel/'+model_judge))  # 保存したモデルをロード
    model.eval()  # 推論モードに設定
    # 推論実行
    with torch.no_grad():  # 勾配計算を無効化
        output = model(img_tensor)
        result = output.argmax(dim=1)  # 最大値を持つインデックスを取得
    return result_list[result.item()]

#######----数式判定
def IMAGE_TO_LATEX(jpg):
    from pix2tex import cli as pix2tex
    from PIL import Image
    from munch import Munch
    arguments = Munch({'config': "/Users/hasegawatakahiro/Desktop/EasyHandWritten/EasyHandWrittenAPI/studiedModel/config.yaml",
                        'checkpoint': "/Users/hasegawatakahiro/Desktop/EasyHandWritten/EasyHandWrittenAPI/studiedModel/mixed_e20_step580.pth", 'no_cuda': True, 'no_resize': False})
    model = pix2tex.LatexOCR(arguments)
    img = Image.open(jpg)
    math = model(img)
    return math
# #####---0.AI採点するためのコード
# def resize_pic(pic1):
#     #調整後サイズを指定(横幅、縦高さ)
#     size=(28,28)
#     base_pic=np.zeros((size[1],size[0]),np.uint8)
# #     pic1=cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
#     h,w=pic1.shape[:2]
#     ash=size[1]/h
#     asw=size[0]/w
#     if asw<ash:
#         sizeas=(int(w*asw),int(h*asw))
#     else:
#         sizeas=(int(w*ash),int(h*ash))
#     pic1 = cv2.resize(pic1,dsize=sizeas)
#     base_pic[int(size[1]/2-sizeas[1]/2):int(size[1]/2+sizeas[1]/2),int(size[0]/2-sizeas[0]/2):int(size[0]/2+sizeas[0]/2)]=pic1
#     return base_pic


# #######----２桁数字判定
# def IMAGE_TO_PREDICT(BMP):
#     image_width=28
#     image_height=28
#     color_setting=1
#     trim_lists=[]
#     num_lists = []
#     img = cv2.imread(BMP)
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # 軽く膨張処理
# #     img_gray = cv2.dilate(img_gray, None, iterations = 1)
#     contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for rect in contours:
#     # 検出できた個数
#     #     print(cv2.contourArea(rect))
#         if cv2.contourArea(rect) > 10:
#     #余分なものを取り除いた個数
#     #         print(cv2.contourArea(rect))
#             x1, y1, w, h = cv2.boundingRect(rect)
#             if x1!=0:
#                 x1=x1-1
#             x2=x1+w+2
#             if y1!=0:
#                 y1 = y1-1
#             y2=y1+h+2
#     # 高さが下すぎるところからはじまるもの（y1）、高さが上すぎるところで終わるもの（y2）は省く 
#             if y1 < img_gray.shape[0]-10 and y2 > 10 and y2-y1 > 10:
#                 trim_lists.append([x1,x2,y1,y2])
#     trim_lists.sort()
#     for trim_list_i in trim_lists:
#         x1,x2,y1,y2 = trim_list_i
#         y3 = y2 if y1 > y2 else y1
#         y4 = y1 if y2 < y1 else y2
#         y1 = y3
#         y2 = y4
#         img = cv2.imread(BMP)
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     #軽く膨張処理
#         img_test = img_gray[y1:y2, x1:x2] 
#     # 以下、mnistのデータセットに合わせて加工した
#         img_re = resize_pic(img_test)# これが保存される
#         img_resh = img_re.reshape(image_width, image_height, color_setting).astype('float32')/255
#         model= load_model('saitenMac/number.h5')
#         prediction = model.predict(np.array([img_resh]))
#         result = prediction[0]
#         num = result.argmax()
# ####################### 結果の確認  
#         num_lists.append(num)
#         # 画像と結果の照らし合わせ
#         pre_ans=""
#         for num in num_lists:
#             pre_ans += str(num)
#     return pre_ans

