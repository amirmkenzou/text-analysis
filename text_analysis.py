import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from hazm import Normalizer, word_tokenize, stopwords_list
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
import arabic_reshaper
from bidi.algorithm import get_display
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(page_title="Persian Text Analysis", layout="wide")

# تابع برای درست کردن متن فارسی
def fix_persian_text(text):
    if text is None:
        return ""
    text = str(text)
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return bidi_text

# Custom CSS
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/npm/vazirmatn@33.003/Vazirmatn-font-face.css');
    
    * {
        font-family: Vazirmatn, sans-serif !important;
    }
    
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        direction: rtl;
    }
    
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        background-color: white;
        color: black;
        direction: rtl;
        text-align: right;
    }
    
    .stButton > button {
        background-color: white;
        color: black;
        border-radius: 8px;
    }
    
    div[data-testid="stDataFrame"] {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("📝 پردازش و تحلیل متن فارسی - پروژه ۴")
st.markdown("---")

default_texts = [
    "هوش مصنوعی آینده تکنولوژی را متحول خواهد کرد",
    "یادگیری ماشین شاخه‌ای از علم داده و هوش مصنوعی است",
    "پردازش زبان طبیعی به کامپیوتر کمک می‌کند تا زبان انسان را درک کند",
    "شبکه‌های عصبی از مغز انسان الهام گرفته شده‌اند",
    "علم داده ترکیبی از آمار، برنامه‌نویسی و دانش تخصصی است",
    "الگوریتم‌های یادگیری عمیق در تشخیص تصویر بسیار موفق هستند",
    "تحلیل احساسات می‌تواند نظرات مشتریان را بررسی کند",
    "مدل‌های زبانی بزرگ قابلیت تولید متن دارند",
    "داده‌کاوی به کشف الگوهای پنهان در داده‌ها می‌پردازد",
    "یادگیری تقویتی برای بازی‌های رایانه‌ای کاربرد دارد"
]

st.subheader("📥 ورودی متن‌های فارسی")

input_method = st.radio(
    "روش ورودی:",
    ["استفاده از متن‌های نمونه", "آپلود فایل CSV", "ورود دستی متن‌ها"],
    horizontal=True
)

texts = []

if input_method == "استفاده از متن‌های نمونه":
    texts = default_texts
    st.success(f"✅ {len(texts)} متن نمونه بارگذاری شد")
    with st.expander("نمایش متن‌های نمونه"):
        for i, text in enumerate(texts, 1):
            st.write(f"{i}. {text}")

elif input_method == "آپلود فایل CSV":
    st.info("فایل CSV باید یک ستون با نام 'text' داشته باشد")
    uploaded_file = st.file_uploader("فایل CSV را آپلود کنید", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'text' in df.columns:
            texts = df['text'].dropna().tolist()
            st.success(f"✅ {len(texts)} متن از فایل بارگذاری شد")
            st.dataframe(df.head(10))
        else:
            st.error("❌ فایل باید ستونی با نام 'text' داشته باشد")

else:  
    st.info("هر متن را در یک خط جداگانه وارد کنید")
    text_input = st.text_area("متن‌های خود را وارد کنید:", height=200)
    
    if text_input.strip():
        texts = [line.strip() for line in text_input.split('\n') if line.strip()]
        st.success(f"✅ {len(texts)} متن وارد شد")

if texts and len(texts) >= 2:
    st.markdown("---")
    
    # Preprocessing
    st.subheader("🔧 پیش‌پردازش متن")
    
    normalizer = Normalizer()
    persian_stopwords = set(stopwords_list())
    
    def preprocess_text(text, remove_stopwords=True):
        
        text = normalizer.normalize(text)

        tokens = word_tokenize(text)
        if remove_stopwords:
            tokens = [t for t in tokens if t not in persian_stopwords and len(t) > 1]
        return ' '.join(tokens)
    
    col1, col2 = st.columns(2)
    with col1:
        remove_stopwords = st.checkbox("حذف کلمات ایست (Stopwords)", value=True)
    with col2:
        show_preprocessing = st.checkbox("نمایش نتیجه پیش‌پردازش", value=False)
    
    processed_texts = [preprocess_text(text, remove_stopwords) for text in texts]
    
    if show_preprocessing:
        with st.expander("مقایسه متن اصلی و پردازش‌شده"):
            comparison_df = pd.DataFrame({
                'متن اصلی': texts[:5],
                'متن پردازش‌شده': processed_texts[:5]
            })
            st.dataframe(comparison_df)
    
    st.markdown("---")
    st.subheader("⚙️ تنظیمات روش‌های Embedding")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        use_tfidf = st.checkbox("✅ TF-IDF", value=True)
        if use_tfidf:
            tfidf_max_features = st.slider("تعداد ویژگی TF-IDF", 10, 500, 100)
    
    with col2:
        use_word2vec = st.checkbox("✅ Word2Vec", value=True)
        if use_word2vec:
            w2v_size = st.slider("بعد Word2Vec", 50, 300, 100, 50)
    
    with col3:
        use_transformer = st.checkbox("✅ Sentence Transformer", value=True)
        if use_transformer:
            transformer_model = st.selectbox(
                "مدل Transformer:",
                ["paraphrase-multilingual-MiniLM-L12-v2", 
                 "paraphrase-multilingual-mpnet-base-v2"]
            )
    
    st.markdown("---")
    st.subheader("📉 تنظیمات کاهش بعد (SVD)")
    
    apply_svd = st.checkbox("اعمال SVD برای کاهش بعد", value=True)
    if apply_svd:
        n_components = st.slider("تعداد کامپوننت‌های SVD", 2, 50, 10)
    
    if st.button("🚀 شروع تحلیل و مقایسه"):
        results = {}
        
        with st.spinner("در حال پردازش..."):
            
            # ==================== TF-IDF ====================
            if use_tfidf:
                st.markdown("---")
                st.subheader("📊 روش 1: TF-IDF")
                
                vectorizer = TfidfVectorizer(max_features=tfidf_max_features)
                tfidf_matrix = vectorizer.fit_transform(processed_texts).toarray()
                
                st.write(f"**شکل ماتریس:** {tfidf_matrix.shape}")
                st.write(f"**تعداد ویژگی‌ها:** {len(vectorizer.get_feature_names_out())}")
                
                # SVD
                if apply_svd and n_components < tfidf_matrix.shape[1]:
                    svd = TruncatedSVD(n_components=n_components, random_state=42)
                    tfidf_reduced = svd.fit_transform(tfidf_matrix)
                    
                    st.write(f"**شکل پس از SVD:** {tfidf_reduced.shape}")
                    st.write(f"**واریانس توضیح داده شده:** {svd.explained_variance_ratio_.sum():.2%}")
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(np.cumsum(svd.explained_variance_ratio_), marker='o')
                    ax.set_xlabel(fix_persian_text('تعداد کامپوننت'))
                    ax.set_ylabel(fix_persian_text('واریانس تجمعی'))
                    ax.set_title(fix_persian_text('واریانس توضیح داده شده توسط SVD - TF-IDF'))
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    results['TF-IDF'] = tfidf_reduced
                else:
                    results['TF-IDF'] = tfidf_matrix
                
                feature_names = vectorizer.get_feature_names_out()
                top_indices = np.argsort(tfidf_matrix.sum(axis=0))[-10:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                top_scores = [tfidf_matrix.sum(axis=0)[i] for i in top_indices]
                
                fig, ax = plt.subplots(figsize=(10, 5))
                fixed_words = [fix_persian_text(w) for w in top_words]
                ax.barh(fixed_words, top_scores, color='steelblue')
                ax.set_xlabel(fix_persian_text('امتیاز TF-IDF'))
                ax.set_title(fix_persian_text('10 کلمه برتر بر اساس TF-IDF'))
                plt.tight_layout()
                st.pyplot(fig)
            
            # ==================== Word2Vec ====================
            if use_word2vec:
                st.markdown("---")
                st.subheader("📊 روش 2: Word2Vec (Average Embedding)")
                
                try:
                    with st.spinner("بارگذاری مدل Word2Vec..."):

                        from gensim.models import Word2Vec
                        
                        sentences = [word_tokenize(normalizer.normalize(text)) for text in texts]
                        w2v_model = Word2Vec(sentences=sentences, vector_size=w2v_size, 
                                           window=5, min_count=1, workers=4, epochs=10)
                        
                        st.success("✅ مدل Word2Vec آموزش داده شد")
                        st.write(f"**تعداد کلمات در واژگان:** {len(w2v_model.wv)}")
                        
                        def get_sentence_embedding(sentence, model):
                            tokens = word_tokenize(normalizer.normalize(sentence))
                            vectors = [model.wv[word] for word in tokens if word in model.wv]
                            if vectors:
                                return np.mean(vectors, axis=0)
                            else:
                                return np.zeros(model.vector_size)
                        
                        w2v_matrix = np.array([get_sentence_embedding(text, w2v_model) for text in texts])
                        
                        st.write(f"**شکل ماتریس:** {w2v_matrix.shape}")
                        
                        # SVD
                        if apply_svd and n_components < w2v_matrix.shape[1]:
                            svd = TruncatedSVD(n_components=n_components, random_state=42)
                            w2v_reduced = svd.fit_transform(w2v_matrix)
                            st.write(f"**شکل پس از SVD:** {w2v_reduced.shape}")
                            st.write(f"**واریانس توضیح داده شده:** {svd.explained_variance_ratio_.sum():.2%}")
                            results['Word2Vec'] = w2v_reduced
                        else:
                            results['Word2Vec'] = w2v_matrix
                        
                        if len(w2v_model.wv) > 0:
                            sample_word = list(w2v_model.wv.index_to_key)[0]
                            similar = w2v_model.wv.most_similar(sample_word, topn=5)
                            
                            with st.expander(f"کلمات مشابه به '{sample_word}'"):
                                for word, score in similar:
                                    st.write(f"- **{word}**: {score:.3f}")
                
                except Exception as e:
                    st.error(f"❌ خطا در Word2Vec: {str(e)}")
            
            # ==================== Sentence Transformer ====================
            if use_transformer:
                st.markdown("---")
                st.subheader("📊 روش 3: Sentence Transformer (Pre-trained)")
                
                try:
                    with st.spinner(f"بارگذاری مدل {transformer_model}..."):
                        model = SentenceTransformer(transformer_model)
                        
                        transformer_matrix = model.encode(texts, show_progress_bar=False)
                        
                        st.success("✅ Embedding با Transformer انجام شد")
                        st.write(f"**شکل ماتریس:** {transformer_matrix.shape}")
                        
                        # SVD
                        if apply_svd and n_components < transformer_matrix.shape[1]:
                            svd = TruncatedSVD(n_components=n_components, random_state=42)
                            transformer_reduced = svd.fit_transform(transformer_matrix)
                            st.write(f"**شکل پس از SVD:** {transformer_reduced.shape}")
                            st.write(f"**واریانس توضیح داده شده:** {svd.explained_variance_ratio_.sum():.2%}")
                            results['Transformer'] = transformer_reduced
                        else:
                            results['Transformer'] = transformer_matrix
                
                except Exception as e:
                    st.error(f"❌ خطا در Transformer: {str(e)}")
            
            if results:
                st.markdown("---")
                st.subheader("🔍 محاسبه شباهت معنایی بین جملات")
                
                for method_name, embeddings in results.items():
                    st.markdown(f"### روش: {method_name}")
                    
                    similarity_matrix = cosine_similarity(embeddings)
                    
                    # Heatmap
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # لیبل‌ها (نمایش 30 کاراکتر اول هر متن)
                    labels = [fix_persian_text(text[:30] + '...') for text in texts]
                    
                    sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                               xticklabels=labels, yticklabels=labels, ax=ax,
                               cbar_kws={'label': fix_persian_text('شباهت')})
                    ax.set_title(fix_persian_text(f'ماتریس شباهت معنایی - {method_name}'))
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # یافتن مشابه‌ترین جفت‌ها
                    np.fill_diagonal(similarity_matrix, -1)  # حذف diagonal
                    most_similar_idx = np.unravel_index(similarity_matrix.argmax(), similarity_matrix.shape)
                    most_similar_score = similarity_matrix[most_similar_idx]
                    
                    st.success(f"**مشابه‌ترین جفت (امتیاز: {most_similar_score:.3f}):**")
                    st.write(f"📄 متن 1: {texts[most_similar_idx[0]]}")
                    st.write(f"📄 متن 2: {texts[most_similar_idx[1]]}")
                    
                    st.markdown("---")
                
                st.markdown("---")
                st.subheader("📊 مقایسه روش‌های مختلف")
                
                comparison_data = []
                for method_name, embeddings in results.items():
                    similarity_matrix = cosine_similarity(embeddings)
                    np.fill_diagonal(similarity_matrix, 0)
                    
                    comparison_data.append({
                        'روش': method_name,
                        'ابعاد': embeddings.shape[1],
                        'میانگین شباهت': similarity_matrix.mean(),
                        'انحراف معیار شباهت': similarity_matrix.std(),
                        'حداکثر شباهت': similarity_matrix.max(),
                        'حداقل شباهت': similarity_matrix.min()
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                ax = axes[0]
                x_pos = np.arange(len(comparison_df))
                fixed_methods = [fix_persian_text(m) for m in comparison_df['روش']]
                
                ax.bar(x_pos, comparison_df['میانگین شباهت'], yerr=comparison_df['انحراف معیار شباهت'],
                       capsize=5, color='steelblue', alpha=0.7)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(fixed_methods)
                ax.set_ylabel(fix_persian_text('میانگین شباهت'))
                ax.set_title(fix_persian_text('مقایسه میانگین شباهت روش‌ها'))
                ax.grid(True, alpha=0.3, axis='y')
                
                ax = axes[1]
                methods = comparison_df['روش'].values
                for i, method in enumerate(methods):
                    row = comparison_df[comparison_df['روش'] == method].iloc[0]
                    fixed_method = fix_persian_text(method)
                    ax.plot([row['حداقل شباهت'], row['حداکثر شباهت']], [i, i], 
                           marker='o', linewidth=3, label=fixed_method)
                
                ax.set_yticks(range(len(methods)))
                ax.set_yticklabels(fixed_methods)
                ax.set_xlabel(fix_persian_text('محدوده شباهت'))
                ax.set_title(fix_persian_text('محدوده (Min-Max) شباهت'))
                ax.grid(True, alpha=0.3, axis='x')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # ==================== Clustering Dendrogram ====================
                st.markdown("---")
                st.subheader("🌳 Hierarchical Clustering (Dendrogram)")
                
                selected_method = st.selectbox("انتخاب روش برای Clustering:", list(results.keys()))
                
                embeddings = results[selected_method]
                linkage_matrix = linkage(embeddings, method='ward')
                
                fig, ax = plt.subplots(figsize=(14, 7))
                labels = [fix_persian_text(text[:40] + '...') for text in texts]
                dendrogram(linkage_matrix, labels=labels, ax=ax, leaf_rotation=45, leaf_font_size=9)
                ax.set_title(fix_persian_text(f'Dendrogram - {selected_method}'))
                ax.set_xlabel(fix_persian_text('شماره متن'))
                ax.set_ylabel(fix_persian_text('فاصله'))
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("---")
                st.subheader("📋 نتیجه‌گیری و توصیه‌ها")
                
                best_method_idx = comparison_df['میانگین شباهت'].idxmax()
                best_method = comparison_df.loc[best_method_idx, 'روش']
                
                st.markdown(f"""
                ### تحلیل نتایج:
                
                #### 🥇 بهترین روش بر اساس میانگین شباهت: **{best_method}**
                
                #### مقایسه روش‌ها:
                
                1. **TF-IDF:**
                   - ✅ سریع و کارآمد
                   - ✅ قابل تفسیر (می‌توان اهمیت هر کلمه را دید)
                   - ❌ وابسته به فرکانس، معنای کلمات را درک نمی‌کند
                   - ❌ برای جملات کوتاه یا متن‌های مشابه معنایی ضعیف است
                   - **کاربرد:** جستجو، دسته‌بندی سند، تحلیل موضوعی
                
                2. **Word2Vec:**
                   - ✅ معنای کلمات را یاد می‌گیرد
                   - ✅ کلمات مشابه را تشخیص می‌دهد
                   - ❌ نیاز به corpus بزرگ برای آموزش
                   - ❌ average embedding ممکن است اطلاعات مهم را از دست بدهد
                   - **کاربرد:** تشخیص کلمات مشابه، analogies، جاگذاری کلمات
                
                3. **Sentence Transformer:**
                   - ✅ معنای کل جمله را درک می‌کند
                   - ✅ از مدل‌های pre-trained قدرتمند استفاده می‌کند
                   - ✅ برای شباهت معنایی عالی است
                   - ❌ سنگین‌تر و کندتر
                   - ❌ نیاز به منابع بیشتر
                   - **کاربرد:** جستجوی معنایی، سیستم‌های سوال-جواب، تشابه متنی
                
                ### توصیه نهایی:
                - برای **سرعت و سادگی**: TF-IDF
                - برای **تحلیل کلمات**: Word2Vec
                - برای **شباهت معنایی دقیق**: Sentence Transformer ⭐
                """)

else:
    st.info("👆 لطفاً حداقل 2 متن وارد کنید تا تحلیل آغاز شود")

st.markdown("---")
st.markdown("""
### 💡 ایده‌های خلاقانه پیاده‌سازی شده:
1. ✅ مقایسه سه روش مختلف embedding (TF-IDF, Word2Vec, Transformer)
2. ✅ کاهش بعد با SVD و نمایش واریانس
3. ✅ Heatmap شباهت معنایی برای هر روش
4. ✅ Hierarchical Clustering با Dendrogram
5. ✅ مقایسه کمی و کیفی روش‌ها
6. ✅ نمایش Top Words در TF-IDF
7. ✅ نمایش Similar Words در Word2Vec
8. ✅ پشتیبانی از چند منبع ورودی (نمونه، CSV، دستی)
9. ✅ پیش‌پردازش متن فارسی با Hazm
10. ✅ رابط کاربری تعاملی و زیبا با Streamlit
""")
