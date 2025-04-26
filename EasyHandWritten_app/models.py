from django.db import models
from django.db.models.signals import post_migrate
from django.dispatch import receiver

class Judge(models.Model):
    judge_str = models.CharField(max_length=255)

    def __str__(self):
        return self.judge_str

# マイグレーション後にデータを挿入する処理
@receiver(post_migrate)
def insert_initial_data(sender, **kwargs):
    # 初期データを挿入
    if not Judge.objects.exists():  # データがなければ挿入
        Judge.objects.bulk_create([
            Judge(judge_str='数字判定'),
            Judge(judge_str='ひらがな判定'),
            Judge(judge_str='マルバツ判定'),
            Judge(judge_str='数式判定')
        ])

class HandWritten(models.Model):
    file_name = models.TextField()
    judge_id = models.IntegerField(default=1)
    image = models.BinaryField(null=True)
    result = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.file_name