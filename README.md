# Image Captioning با DenseNet + BERT + LSTM (PyTorch)

## ایده اصلی
این پروژه یک مدل Image Captioning پیاده‌سازی کرده که با ترکیب ویژگی‌های تصویری و زبانی، کپشن متنی تولید می‌کند.  
معماری از **DenseNet برای تصویر** و **BERT Embedding + LSTM Decoder** برای متن استفاده می‌کند.

---

## معماری مدل

Image
  ↓
  
DenseNet121 (pretrained، حذف لایه طبقه‌بندی)

  ↓
  
ویژگی تصویر (1024 بعد)
  ↓
  
Linear Projection → Image Embedding (768 بعد)

                ┐
                
Text Tokens → BERT Embedding (768 بعد)

                ├── Concatenate (1536 بعد)
                
                ↓
                
          Uni-directional LSTM
          
                ↓
                
      Residual Connection
      
                ↓
                
        Linear + Softmax → پیش‌بینی توکن بعدی

### اجزای کلیدی
- **DenseNet121**: استخراج ویژگی تصویری
- **Linear Projection**: نگاشت ویژگی تصویر به فضای embedding
- **BERT Embedding**: نمایش زبانی توکن‌ها
- **Uni-LSTM Decoder**: تولید توالی کلمات
- **Residual Path**: بهبود جریان گرادیان و پایداری آموزش

---

## Dataset و Tokenization
- استفاده از **HuggingFace BertTokenizer**
- توکن‌های استاندارد: `[CLS]`, `[SEP]`, `[PAD]`
- شیفت توکن‌ها برای sequence prediction
- خروجی Dataset شامل:
  - `image_features`
  - `input_ids`
  - `attention_mask`
  - `labels`

---

## آموزش مدل
- Optimizer: AdamW  
- Loss: CrossEntropy (ignore PAD)  
- GPU: Tesla T4  
- زمان آموزش: ~21 دقیقه  
- Loss از ~4.7 به ~3.0 کاهش یافته است

---

## تولید کپشن (Inference)
- شروع با `[CLS]`
- تولید توکن به صورت مرحله‌ای
- توقف در `[SEP]`
- پشتیبانی از **Greedy** و **Top-K Sampling**

---

## نتیجه
مدل قادر است کپشن‌های **معنادار و طبیعی** تولید کند و معماری آن به راحتی قابل توسعه به **Transformer Decoder** یا **Cross-Attention** است.
