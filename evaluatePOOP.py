import serial, torch, os
from torch import nn as nn
from torchvision import transforms, models
from PIL import Image
import time
# install pyserial, torch, torchvision, pillow

pyserial = serial.Serial(
    port = 'COM4',
    baudrate = 9600
)

class CustomModel(nn.Module):
    def __init__(self, drop_prop, num_classes):
        super(CustomModel, self).__init__()

        # Load the pretrained model from pytorch
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Configure Classifier
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(drop_prop),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
            x = self.model(x)
            return x

def eval(filename : str) :
    img = Image.open(filename)
    img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.599459, 0.549111, 0.437853]
            , [0.170636, 0.201447, 0.246873])
    ])(img).unsqueeze(0).to(device)

    pred = model(img)
    pred = pred.argmax(dim=1)
    return int(str(pred.item()))


if __name__ == '__main__':
    CURRENT_STATE = "STAND_BY"
    NEXT_STATE = CURRENT_STATE
    FSR = 0
    US = 0
    FSR_duration = 3
    PIC_wait = 3
    IDEAL = 6
    NORMAL = 9

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CustomModel(0.5, 4).to(device)
    model.load_state_dict(torch.load('ACC956_model_drop5_lr00007.pth', map_location=device))
    model.eval()

    while CURRENT_STATE != "FINISH":
        if pyserial.readable():
            response = str(pyserial.readline()).strip("b'").strip("\\r\\n'")
            response = response.split(",")
            FSR, US = int(response[0]), float(response[1])
            print(f"FSR = {FSR}, US = {US}")
            FSR = 1 if FSR > 3 else 0
            US = 1 if US < 1500 else 0

        if CURRENT_STATE == "STAND_BY":
            is_pooped = False
            if FSR == 1 :
                start_time = time.time()
                NEXT_STATE = "SIT_IN"

        if CURRENT_STATE == "SIT_IN":
            if FSR == 0:
                if time.time() - start_time > FSR_duration :
                    if is_pooped :
                        NEXT_STATE = "RESULT"
                    else :
                        NEXT_STATE = "STAND_BY"
            
            if US == 1:
                poo_time = time.time() # 똥 싸기 시작한 시간
                NEXT_STATE = "STAND_PIC"

        if CURRENT_STATE == "STAND_PIC" :
            if US == 1 :
                poo_time = time.time()
            else:
                if time.time() - poo_time > PIC_wait :
                    # ############################
                    # ****** 여기서 찰칵 *********
                    # ############################

                    if not is_pooped:
                        is_pooped = eval('test.jpg')
                        if is_pooped :
                            first_poo_time = time.time() # 똥 다 싼 시간
                    NEXT_STATE = "SIT_IN"
    

        if CURRENT_STATE == "RESULT" :
            first_poo_time = first_poo_time - start_time
            after_poo_time = time.time() - poo_time
            NEXT_STATE = "STAND_BY"


            if first_poo_time < IDEAL * 60 :
                constipation_rate = 0
            elif first_poo_time < NORMAL * 60 :
                constipation_rate = 1
            else :
                constipation_rate = 2
            
            if after_poo_time < 5*60 :
                bad_habbit = 0
            else :
                bad_habbit = 1

            if is_pooped :
                stool = eval('test.jpg')
                print(f"{stool},{constipation_rate},{bad_habbit}")
                NEXT_STATE = "FINISH"
                break
            
        print(f"FSR = {FSR}, US = {US}, CURRENT_STATE = {CURRENT_STATE}") # and this line

        time.sleep(0.001)
        os.system("cls") # delete this line in practice
        CURRENT_STATE = NEXT_STATE

# 총 앉아있는 시간 =현재시각 - 감압센서 on 된 시각


#entire_time = datetime.timedelta(days=1)


# 똥시작타이머 = 현재 시각 - 똥 시작 시각

# 앉아서 변이 나오는데까지 걸리는 시간 = 똥 시작 시각 - 감압센서 on 된 시각

# 배변 활동 끝난 후 앉아있는 시간 = 감압센서 off된 시각 - 똥 시작 시각

# 안좋은 습관 = bool

# 감압센서 평균값 = 
#        지난 5초간 감압센서 평균값

# if 감압센서bool값 == True :
#     총 앉아있는 시간 시작
#     루프(
#         if 똥시작타이머 <= 5분 :
#             루프 (
#                 if 초음파센서bool값 == True :
#                     3초 딜레이
#                     카메라 on
#                     방금 받은 이미지 분석 시작
#                     if 방금 받은 이미지 분석 == 똥 :
#                         똥시작타이머
#                         앉아서 변이 나오는데까지 걸리는 시간 = 총 앉아있는 시간 현재값
#                         가져갈 이미지 데이터 = 방금 받은 이미지 데이터
#                         루프 탈출
#                     else 이미지 분석 != 똥:
#                         루프탈출

#                 else 초음파센서bool값 == False :
#                     루프탈출
#              루프탈출 
#             )
#         if 똥시작타이머 >5분 :
#             안좋은 습관 = True
#             루프 (
#                 if 초음파센서bool값 == True :
#                     3초 딜레이
#                     카메라 on
#                     방금 받은 이미지 분석 시작
#                     if 방금 받은 이미지 분석 == 똥 :
#                         똥시작타이머
#                         앉아서 변이 나오는데까지 걸리는 시간 = 총 앉아있는 시간 현재값
#                         가져갈 이미지 데이터 = 방금 받은 이미지 데이터
#                         루프 탈출
#                     else 이미지 분석 != 똥:
#                         루프탈출

#                 else 초음파센서bool값 == False :
#                     루프탈출
#              루프탈출 
#             )
#         if 감압센서 평균값 False :
#            루프탈출
#     )

# else :
#     총 앉아있는 시간 종료

#     똥시작타이머 종료
#     배변 활동 끝난 후 앉아있는 시간 = 총 앉아있는 시간 - 똥시작타이머
#     if 총 앉아있는 시간 < 6분:
#         습관 = 정상
#     elif 총 앉아있는 시간 <9분 :
#         습관 = 주의
#     else :
#         습관 = 경고



# 보낼거

# 가져갈 이미지 데이터(정상, 설사, empty 중 하나) (str)
# 습관 (str)
# 안 좋은 습관 (bool 아니면 str)




