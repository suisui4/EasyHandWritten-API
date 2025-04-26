from django.core.paginator import Paginator, Page
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response

from EasyHandWritten_app import utils
from EasyHandWritten_app.models import HandWritten

# 画像をアップロードされたものを判定し、DBに登録結果を返す
@api_view(['POST'])
def upload_image(request):
    file = request.FILES.get('file')
    file_name = request.POST.get('file_name')
    judge_id = request.POST.get('judge_id')
    image_data = file.read()
    # ファイルをバイナリデータに変換してDBに保存
    if  judge_id == '4':
        handwritten = HandWritten(
            file_name=file_name,
            judge_id=int(judge_id),
            image=image_data,
            result=utils.IMAGE_TO_LATEX(file)
            )
    else:
        handwritten = HandWritten(
            file_name=file_name,
            judge_id=int(judge_id),
            image=image_data,
            result=utils.IMAGE_TO_RESULT(image_data, judge_id)
        )
    handwritten.save()
    return JsonResponse({"imageId":handwritten.id}, status=201)

# 判定する項目の名前を返す
@api_view(['GET'])
def get_AIname(request, judge_id):
    AIname_dict = utils.get_category_names()
    return Response({"judge_name": AIname_dict.get(judge_id)})


# 履歴のデータを返す
@api_view(['GET'])
def get_latest_image(request):
    # クエリパラメータからページ番号とページサイズを取得（デフォルトは1ページ目、10件）
    page = int(request.GET.get('page', 1))
    page_size = int(request.GET.get('pageSize', 10))
    
    # 保存してあるデータを取得
    history_data = HandWritten.objects.all()
    # ページネーション処理
    paginator = Paginator(history_data, page_size)
    page_data = paginator.get_page(page)
    
    # AI判定名を取得
    AIname_dict = utils.get_category_names()
    
    # 画像保存リスト
    images_data = []
    for entry in page_data:
        # 画像を Base64 に変換
        image_base64 = utils.get_image(entry.image)
        
        # レスポンスのデータをリストに追加
        images_data.append({
            "id": entry.id,
            "image": image_base64,
            "result": entry.result,
            "category": AIname_dict.get(entry.judge_id)
        })
    
    # ページ情報をレスポンスに追加
    response_data = {
        "images": images_data,
        "totalCount": paginator.count,  # 総アイテム数
        "page": page,  # 現在のページ番号
        "pageSize": page_size,  # ページサイズ
        "totalPages": paginator.num_pages,  # 総ページ数
    }
    
    return JsonResponse(response_data)

# DBから削除
@api_view(['DELETE'])
def delete_history(request, id):
    entry = HandWritten.objects.get(id=id)
    entry.delete()
    return Response({"message": "Deleted successfully"}, status=200)

# idを指定して画像データと結果を取得
@api_view(['GET'])
def get_image_by_id(request, image_id):
    image = HandWritten.objects.get(id=image_id)
    image_data = {
    'file_name': image.file_name,
    'image': utils.get_image(image.image),
    'result': image.result
    }
    return Response(image_data)