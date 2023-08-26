from django.shortcuts import render
from .models import Cluster
from .forms import TaskForm
import numpy as np
import random
import matplotlib.pyplot as plt
import io, base64
import pickle
from django.shortcuts import get_object_or_404, redirect
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import pandas as pd


def k_means_clustering(X, clstr):

    kmeans = KMeans(n_clusters=clstr)
    kmeans.fit(X)
    labels = kmeans.labels_

    W = [[] for _ in range(clstr)]
    for i, label in enumerate(labels):
        W[label].append(X[i])

    centroids = [np.mean(cluster, axis=0) for cluster in W]
    return W, centroids


def clustering(K, n, lower_bound, upper_bound, metrika, q1, q2, q3):
    q1, q2, q3 = np.array(q1), np.array(q2), np.array(q3)
    W = [[q1], [q2], [q3]]
    metrics_dict = {
        1: lambda a, b: np.sum(np.abs(a - b)),
        2: lambda a, b: np.sqrt(np.sum(np.power(a - b, 2))),
        3: lambda a, b: np.max(np.abs(a - b))
    }
    for i in range(K):
        x = np.random.uniform(low=lower_bound, high=upper_bound, size=n)
        distances = [
            metrics_dict[metrika](q1, x),
            metrics_dict[metrika](q2, x),
            metrics_dict[metrika](q3, x)
        ]
        Min = np.argmin(distances)

        if distances.count(distances[Min]) > 1:
            Min = random.choice([i for i in range(3) if distances[i] == distances[Min]])

        W[Min].append(x)
    return W


def hierar_clustering(linkage, affinity, X, clstr):
    clustering = AgglomerativeClustering(
        n_clusters=clstr, linkage=linkage, affinity=affinity
    )
    labels = clustering.fit_predict(X)
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(X[i])
    W = list(clusters.values())
    return W


def clustering_view(request):
    if request.method == 'POST':
        algorithm = request.POST.get('algorithm')
        W = None
        base64_data = None
        if algorithm == 'users':
            name_data = str(request.POST.get('name_data_us'))
            clstr = 3
            K = int(request.POST.get('K_us'))
            n = int(request.POST.get('n_us'))
            lower_bound = int(request.POST.get('lower_bound_us'))
            upper_bound = int(request.POST.get('upper_bound_us'))
            metrika = int(request.POST.get('metrika'))

            q1 = list(map(int, request.POST.get('q1').translate({ord(i): None for i in (']', '[', ',')}).split()))
            q2 = list(map(int, request.POST.get('q2').translate({ord(i): None for i in (']', '[', ',')}).split()))
            q3 = list(map(int, request.POST.get('q3').translate({ord(i): None for i in (']', '[', ',')}).split()))

            W = clustering(K, n, lower_bound, upper_bound, metrika, q1, q2, q3)
            w_bytes = pickle.dumps(W)

            user = None
            if request.user.is_authenticated:
                user = request.user

            cluster = Cluster(user=user,name_data=name_data, k=K, n=n, lower_bound=lower_bound, upper_bound=upper_bound,
                              metrika=metrika, q1=q1, q2=q2, q3=q3, w=w_bytes, clstr_count=clstr)
            if request.user.is_authenticated:
                cluster.save()
            base64_data = image_cluster(W, q1, q2, q3, clstr)
        elif algorithm == 'k_means':
            name_data = str(request.POST.get('name_data_me'))
            clstr = int(request.POST.get('clstr_count_me'))
            K = int(request.POST.get('K_me'))
            n = int(request.POST.get('n_me'))
            lower_bound = int(request.POST.get('lower_bound_me'))
            upper_bound = int(request.POST.get('upper_bound_me'))
            metrika = '3'
            X = np.random.uniform(low=lower_bound, high=upper_bound, size=(K, n))
            W, centroids = k_means_clustering(X, clstr)
            w_bytes = pickle.dumps(W)
            user = None
            if request.user.is_authenticated:
                user = request.user
            centroids = list(map(list, centroids))
            q1 = centroids[0]
            q2 = centroids[1]
            q3 = centroids[2]
            cluster = Cluster(user=user, name_data=name_data, k=K, n=n, lower_bound=lower_bound, upper_bound=upper_bound,
                              metrika=metrika, q1=q1, q2=q2, q3=q3, w=w_bytes, clstr_count=clstr)
            if request.user.is_authenticated:
                cluster.save()
            base64_data = image_cluster(W, q1, q2, q3, clstr)
        elif algorithm == 'hierarchical':
            name_data = str(request.POST.get('name_data'))
            clstr = int(request.POST.get('clstr_count'))
            K = int(request.POST.get('K'))
            n = int(request.POST.get('n'))
            lower_bound = int(request.POST.get('lower_bound'))
            upper_bound = int(request.POST.get('upper_bound'))
            X = np.random.uniform(low=lower_bound, high=upper_bound, size=(K, n))
            linkage = request.POST.get('linkage')
            affinity = request.POST.get('affinity')
            W = hierar_clustering(linkage, affinity, X, clstr)
            w_bytes = pickle.dumps(W)
            user = None
            if request.user.is_authenticated:
                user = request.user
            centroids = [np.mean(cluster, axis=0) for cluster in W]
            q1 = centroids[0]
            q2 = centroids[1]
            q3 = centroids[2]
            cluster = Cluster(user=user,name_data=name_data, k=K, n=n, lower_bound=lower_bound, upper_bound=upper_bound,
                              metrika='3', q1=q1, q2=q2, q3=q3, w=w_bytes, clstr_count=clstr)
            if request.user.is_authenticated:
                cluster.save()
            base64_data = image_cluster(W, q1, q2, q3, clstr)

        return render(request, 'main/cluster.html', {'W': W, 'plot_data': base64_data})
    else:
        return render(request, 'main/cluster.html')


def image_cluster(W, q1, q2, q3, clstr, new_obj=''):
    pca = PCA(n_components=2)
    if new_obj != '':
        plt.scatter(q1[0], q1[1], c='red', marker='o')
        plt.scatter(q2[0], q2[1], c='blue', marker='o')
        plt.scatter(q3[0], q3[1], c='green', marker='o')

    if len(W[0][0]) > 2:
        for i in range(len(W)):
            W[i] = pca.fit_transform(W[i])

    for i in range(clstr):
        colors = ['red', 'blue', 'green', 'yellow', 'pink', 'orange']
        if len(W[i]) > 1:
            plt.scatter(np.array(W[i])[:, 0], np.array(W[i])[:, 1], c=colors[i], marker='o')

    plt.legend(loc='best')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Clustering Results')

    if new_obj != '':
        plt.scatter(new_obj[0], new_obj[1], c='black', marker='*')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image_data = buf.getvalue()

    base64_data = base64.b64encode(image_data).decode('utf-8')
    plt.clf()

    return base64_data


def index(request):
    if request.user.is_authenticated:
        user = request.user
        clusters = Cluster.objects.filter(user=user).order_by('id')
        return render(request, 'main/index.html', {'title': 'Главная страница сайта', 'clusters': clusters})
    else:
        return render(request, 'main/index.html', {'title': 'Главная страница сайта', 'clusters': None})


def cluster_data_view(request):
    if request.method == 'POST':
        algorithm = request.POST.get('algorithm')
        users_file = request.FILES['users_file']
        df = pd.read_excel(users_file)
        objs = df['Признаки'].apply(lambda x: np.array(list(map(float, x.split())))).tolist()
        concatenated_array = np.concatenate(objs)
        lower_bound = np.min(concatenated_array)
        upper_bound = np.max(concatenated_array)
        if algorithm == 'k_means':
            name_data = str(request.POST.get('name_data_me'))
            clstr = int(request.POST.get('clstr_count_me'))
            W, centroids = k_means_clustering(objs, clstr)
            w_bytes = pickle.dumps(W)
            user = None
            if request.user.is_authenticated:
                user = request.user
            centroids = list(map(list, centroids))
            q1 = centroids[0]
            q2 = centroids[1]
            q3 = centroids[2]
            cluster = Cluster(user=user, name_data=name_data, k=len(objs), n=len(objs[0]),
                              lower_bound=lower_bound, upper_bound=upper_bound,
                              metrika='3', q1=q1, q2=q2, q3=q3, w=w_bytes, clstr_count=clstr)
            if request.user.is_authenticated:
                cluster.save()
            base64_data = image_cluster(W, q1, q2, q3, clstr)
            return render(request, 'main/cluster_data.html', {'plot_data': base64_data})
        elif algorithm == 'hierarchical':
            name_data = str(request.POST.get('name_data'))
            clstr = int(request.POST.get('clstr_count'))
            linkage = request.POST.get('linkage')
            affinity = request.POST.get('affinity')
            W = hierar_clustering(linkage, affinity, objs, clstr)
            w_bytes = pickle.dumps(W)
            user = None
            if request.user.is_authenticated:
                user = request.user
            centroids = [np.mean(cluster, axis=0) for cluster in W]
            q1 = centroids[0]
            q2 = centroids[1]
            q3 = centroids[2]
            cluster = Cluster(user=user, name_data=name_data, k=len(objs), n=len(objs[0]),
                              lower_bound=lower_bound, upper_bound=upper_bound,
                              metrika='3', q1=q1, q2=q2, q3=q3, w=w_bytes, clstr_count=clstr)
            if request.user.is_authenticated:
                cluster.save()
            base64_data = image_cluster(W, q1, q2, q3, clstr)
            return render(request, 'main/cluster_data.html', {'plot_data': base64_data})
    return render(request, 'main/cluster_data.html')


def about(request):
    return render(request, 'main/about.html')


def create(request):
    error = ''
    if request.method == 'POST':
        form = TaskForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('home')
        else:
            error = 'Форма неверна!'

    form = TaskForm()
    context = {
        'form': form,
        'error': error
    }
    return render(request, 'main/create.html', context)


def recognition_view(request):
    q1 = None
    q2 = None
    q3 = None
    W = None
    name_data = None
    clstr = None
    lower_bound = None
    upper_bound = None
    n = None
    context = {
        'exam_obj': None,
        'lower_bound': None,
        'upper_bound': None,
        'n': None,
        'name_data': None
    }
    if request.GET:
        elem_id = request.GET['id']
        cluster = Cluster.objects.get(id=elem_id)
        request.session['id'] = elem_id
        context = {
            'exam_obj': [0 for _ in range(cluster.n)],
            'lower_bound': cluster.lower_bound,
            'upper_bound': cluster.upper_bound,
            'n': cluster.n,
            'name_data': cluster.name_data
        }
        return render(request, 'main/recognition.html', context)
    if request.method == 'POST':
        elem_id = request.session.get('id')
        cluster = Cluster.objects.get(id=elem_id)
        W = pickle.loads(cluster.w)
        new_obj = list(map(int, request.POST.get('new_obj').translate({ord(i): None for i in (']', '[', ',')}).split()))
        base64_data = image_cluster(W, cluster.q1, cluster.q2, cluster.q3, cluster.clstr_count, np.array(new_obj))
        midle_s = recognition(W, new_obj)
        context['true_cluster'] = midle_s.index(min(midle_s)) + 1
        midle_s = dict(zip([_ + 1 for _ in range(len(midle_s))], midle_s))
        context['midle_s'] = midle_s
        context['new_obj'] = new_obj
        context['plot_data'] = base64_data
        print(midle_s)
        return render(request, 'main/recognition.html', context)
    else:
        return render(request, 'main/recognition.html', context)


def delete_solution(request, solution_id):
    solution = get_object_or_404(Cluster, id=solution_id)
    solution.delete()
    return redirect('home')


def recognition(W, new_obj):
    while len(new_obj) != 2:
        new_obj = [(new_obj[i] + new_obj[i + 1]) / 2 for i in range(len(new_obj) - 1)]
    new_obj = np.array(new_obj)
    midle_s = []
    for i in range(len(W)):
        distances = np.sum(np.abs(W[i] - new_obj), axis=1)
        midle_s.append(np.mean(distances))
    return midle_s


