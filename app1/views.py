from django.shortcuts import render
import joblib

# Create your views here.
def home(request):
    if request.method == 'POST':
        model = joblib.load('./abc.joblib')
        quiz = float(request.POST['quiz'])
        assignment = float(request.POST['assignment'])
        mid = float(request.POST['mid'])
        final = float(request.POST['final'])

        md = model.predict([[quiz, assignment,mid,final]])
        return render(request, 'index.html', {'prediction': md[0]})
    return render(request, "index.html")
