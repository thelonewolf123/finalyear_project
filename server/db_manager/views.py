from django.shortcuts import render

def index(request):
    context = {}
    return render(request,'db_manager/index.html',context)

def single_person(request,id):
    context = {}
    return render(request,'db_manager/single_person.html',context)

def cross_ref(request,id):
    context = {}
    return render(request,'db_manager/cross_ref.html',context)

def add_person(request):
    context = {}
    return render(request,'db_manager/add_person.html',context)
