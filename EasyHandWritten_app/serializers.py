from django.forms import ValidationError
from rest_framework import serializers
from .models import HandWritten, Judge

class JudgeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Judge
        fields = '__all__'

class HandWrittenSerializer(serializers.ModelSerializer):
    judge = JudgeSerializer()

    class Meta:
        model = HandWritten
        fields = '__all__'

    def validate_image(self, value):
        # 画像がバイナリデータであることを確認
        if not value:
            raise ValidationError("画像データが必要です")
        return value
