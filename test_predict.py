import requests
p = r'E:\Major project\DR_Detection_System\dataset_generalist\Cataract\0_left.jpg'
files = {'file': open(p, 'rb')}
try:
    r = requests.post('http://127.0.0.1:8001/predict', files=files, timeout=300)
    print('STATUS', r.status_code)
    print(r.text)
except Exception as e:
    print('ERROR', e)
