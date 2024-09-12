1. Bone Xray Data 다운로드: https://stanfordmlgroup.github.io/competitions/mura/ 에서 데이터를 신청하여 다운로드한다.
2. 데이터 다운로드 후, process_csv.ipynb를 실행하여 train 데이터, validation 데이터의 정보를 처리하여 기록한 bone_xray_train.csv, bone_xray_valid.csv 파일을 생성한다.
3. 이미지넷 weight 없이 학습을 하고 싶으면 train.ipynb를 실행하고 이미지넷 weight 로드 후에 학습을 하고싶으면 train_with_imagenet.ipynb를 실행한다.
4. 학습 이후, inference.ipynb를 실행하여 validation 데이터에 대하여 인퍼런스를 진행하고 다양한 metric들과 graph를 출력한다.
*순서: process_csv.ipynb-->train.ipynb(or train_with_imagenet.ipynb)-->inference.ipynb