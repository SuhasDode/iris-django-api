# predictor/views.py
import json
import numpy as np
import pickle
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            features = np.array(data['features']).reshape(1, -1)
            prediction = int(model.predict(features)[0])
            return JsonResponse({'prediction': prediction})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'message': 'Send POST request.'})
