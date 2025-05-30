import torch
import torch.nn as nn
from functools import wraps
from types import MethodType
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from typing import Dict, Any

# 定义安全类别
SAFETY_CATEGORIES = [
    "NA: None to Apply",
    "harassment", "harassment_threatening",
    "hate", "hate_threatening",
    "illicit", "illicit_violent",
    "self_harm", "self_harm_instructions", "self_harm_intent",
    "sexual", "sexual_minors",
    "violence", "violence_graphic"
]


class Qwen2_5_VLWithSafetyClassifiers(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # 定义安全类别数量
        self.num_categories = len(SAFETY_CATEGORIES)
        self.num_safety_labels = 2  # 二分类: 安全/不安全

        # 获取隐藏层大小
        hidden_size = config.hidden_size

        # 创建四个分类头
        # 1. 文本安全二分类头 - 安全/不安全
        self.text_safety_classifier = nn.Linear(hidden_size, self.num_safety_labels)

        # 2. 文本类别多分类头 - 对应多个安全类别
        self.text_category_classifier = nn.Linear(hidden_size, self.num_categories)

        # 3. 图像安全二分类头 - 安全/不安全
        self.image_safety_classifier = nn.Linear(hidden_size, self.num_safety_labels)

        # 4. 图像类别多分类头 - 对应多个安全类别
        self.image_category_classifier = nn.Linear(hidden_size, self.num_categories)

        # 用于记录各任务的损失
        self.text_safety_loss = 0.0
        self.text_category_loss = 0.0
        self.image_safety_loss = 0.0
        self.image_category_loss = 0.0

        # 任务权重 - 可设置为可学习参数
        self.text_safety_weight = 1.0
        self.text_category_weight = 1.0
        self.image_safety_weight = 1.0
        self.image_category_weight = 1.0

        # 存储所有logits的属性
        self.all_safety_logits = None

        # 初始化分类头的权重
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """初始化分类头权重"""
        initializer_range = getattr(self.config, 'initializer_range', 0.02)

        # 对每个分类器进行更精细的初始化
        for name, classifier in [
            ("text_safety", self.text_safety_classifier),
            ("text_category", self.text_category_classifier),
            ("image_safety", self.image_safety_classifier),
            ("image_category", self.image_category_classifier)
        ]:
            # 使用正态分布初始化权重
            nn.init.normal_(classifier.weight, mean=0.0, std=initializer_range)

            # 将偏置初始化为零
            if classifier.bias is not None:
                nn.init.zeros_(classifier.bias)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 在分类任务中不向父模型传递标签，避免尺寸不匹配的问题
        causal_labels = None
        classification_labels = None

        if labels is not None:
            # 判断是否为分类任务（标签为列表形式）
            if isinstance(labels, torch.Tensor) and len(labels.shape) > 1 and labels.shape[1] > 1:
                # 如果是分类任务，保存标签但不传给父模型
                classification_labels = labels
            else:
                # 如果是正常的语言建模任务，传递标签
                causal_labels = labels

        # 调用原始模型的前向传播，不带分类标签
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=causal_labels,  # 只在语言建模任务中传递标签
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        # 获取最后一层的隐藏状态
        last_hidden_state = outputs.hidden_states[-1]

        # 处理批次大小
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        # 统一使用最后一个token的表示进行特征提取
        text_pooled = last_hidden_state[:, -1]

        # 提取图像表示，同样使用最后一个token的表示
        if pixel_values is not None:
            image_pooled = last_hidden_state[:, -1]
        else:
            image_pooled = None

        # 应用分类头
        logits = None
        loss = None

        # 文本二分类：安全/不安全
        text_safety_logits = self.text_safety_classifier(text_pooled)
        text_category_logits = self.text_category_classifier(text_pooled)

        if image_pooled is not None:
            # 图像分类
            image_safety_logits = self.image_safety_classifier(image_pooled)
            image_category_logits = self.image_category_classifier(image_pooled)
        else:
            image_safety_logits = None
            image_category_logits = None

        # 将所有logits合并为一个列表
        all_logits = [
            text_safety_logits,
            text_category_logits,
            image_safety_logits,
            image_category_logits
        ]

        # 存储所有logits以便外部访问
        self.all_safety_logits = all_logits

        # 对于训练器计算准确率，我们选择第一个logits作为主要logits
        # 因为Swift需要一个单一的张量而不是列表
        main_logits = text_safety_logits

        # 如果有标签，则计算总损失
        if classification_labels is not None:
            # 确保labels是二维张量 [batch_size, 4]
            if len(classification_labels.shape) == 1:
                classification_labels = classification_labels.unsqueeze(0)

            # 分别计算四个分类器的损失，使用适当的损失函数

            # 1. 文本安全分类 - 二分类任务
            loss_fct_binary = nn.CrossEntropyLoss()
            text_safety_loss = loss_fct_binary(text_safety_logits.view(-1, self.num_safety_labels),
                                               classification_labels[:, 0].long())

            # 2. 文本类别分类 - 多分类任务
            loss_fct_multi = nn.CrossEntropyLoss()
            text_category_loss = loss_fct_multi(text_category_logits.view(-1, self.num_categories),
                                                classification_labels[:, 1].long())

            # 3. 图像安全分类 - 二分类任务
            if image_pooled is not None:
                image_safety_loss = loss_fct_binary(image_safety_logits.view(-1, self.num_safety_labels),
                                                    classification_labels[:, 2].long())

                # 4. 图像类别分类 - 多分类任务
                image_category_loss = loss_fct_multi(image_category_logits.view(-1, self.num_categories),
                                                     classification_labels[:, 3].long())
            else:
                image_safety_loss = 0
                image_category_loss = 0

            # 总损失是四个损失的加权和
            # 可以根据任务重要性调整权重
            text_safety_weight = self.text_safety_weight
            text_category_weight = self.text_category_weight
            image_safety_weight = self.image_safety_weight
            image_category_weight = self.image_category_weight

            # 计算加权总损失
            cls_loss = (text_safety_weight * text_safety_loss +
                        text_category_weight * text_category_loss +
                        image_safety_weight * image_safety_loss +
                        image_category_weight * image_category_loss)

            # 记录每个任务的损失，用于监控
            self.text_safety_loss = text_safety_loss.item()
            self.text_category_loss = text_category_loss.item()
            if image_pooled is not None:
                self.image_safety_loss = image_safety_loss.item()
                self.image_category_loss = image_category_loss.item()

            # 合并语言模型损失和分类损失
            if outputs.loss is not None:
                loss = outputs.loss + cls_loss
            else:
                loss = cls_loss
        else:
            # 如果没有分类标签，则只使用语言模型损失
            loss = outputs.loss

        if not return_dict:
            output = (main_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=main_logits,  # 使用主要logits用于训练器计算准确率
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    def get_all_safety_logits(self):
        """获取所有安全分类相关的logits"""
        return self.all_safety_logits

    def get_safety_predictions(self):
        """获取安全相关的预测结果"""
        if self.all_safety_logits is None:
            return None

        # 解包logits
        text_safety_logits, text_category_logits, image_safety_logits, image_category_logits = self.all_safety_logits

        # 获取各分类器的预测
        text_safety_pred = torch.argmax(text_safety_logits, dim=-1)
        text_category_pred = torch.argmax(text_category_logits, dim=-1)

        predictions = {
            "text_safety": text_safety_pred,
            "text_category": text_category_pred,
        }

        # 如果有图像预测结果
        if image_safety_logits is not None:
            image_safety_pred = torch.argmax(image_safety_logits, dim=-1)
            image_category_pred = torch.argmax(image_category_logits, dim=-1)
            predictions.update({
                "image_safety": image_safety_pred,
                "image_category": image_category_pred
            })

        return predictions


def get_model_tokenizer_qwen2_5_vl_safety(model_dir: str, model_info: Any, model_kwargs: Dict[str, Any], load_model: bool = True, **kwargs):
    """自定义函数，加载带有安全分类器的Qwen2.5-VL模型"""
    # 加载处理器
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    # 加载模型
    model = None
    if load_model:
        try:
            print(f"加载带有安全分类器的Qwen2.5-VL模型: {model_dir}")

            # 加载配置
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

            # 获取torch_dtype
            torch_dtype = model_kwargs.get('torch_dtype', None)

            # 加载基础模型
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_dir,
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                **model_kwargs
            )

            # 创建带有安全分类器的模型
            model = Qwen2_5_VLWithSafetyClassifiers(config)

            # 复制基础模型的参数
            missing_keys, unexpected_keys = model.load_state_dict(base_model.state_dict(), strict=False)
            print(f"加载模型状态字典 - 缺少的键: {missing_keys}, 意外的键: {unexpected_keys}")

            # 设置安全类别映射
            model.config.safety_categories = SAFETY_CATEGORIES
            model.config.safety_rating_map = {
                "Safe": 0,
                "Unsafe": 1
            }

            # 设置num_labels和id2label/label2id
            model.config.num_labels = 4  # 四个分类任务

            # 设置安全标签映射
            model.config.id2label = {
                0: "text_safety",
                1: "text_category",
                2: "image_safety",
                3: "image_category"
            }
            model.config.label2id = {v: k for k, v in model.config.id2label.items()}

        except Exception as e:
            print(f"加载模型时出错: {e}")
            raise e

    return model, processor
