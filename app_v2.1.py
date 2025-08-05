import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Load models and scaler
rf = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')

st.markdown("<h1 style='text-align: center;'>Chẩn Đoán Hội Chứng YHCT Bệnh Đái Tháo Đường Type 2</h1>", unsafe_allow_html=True)
st.markdown("========================================================================================")

### Input
with st.form("my_form"):
    left, right = st.columns(2)
    with left:

        st.write("1. CÁC CHỈ SÔ")
        bmi = st.number_input("BMI:", key="bmi_input")
        pulse_frequency = st.number_input("Tần số mạch:", key ="pulse_frequency_input", min_value=0, max_value=200, value=70)
        st.write("2. VỌNG CHẨN")
        emaciation = st.checkbox("Gầy")
        obesity = st.checkbox("Béo bệu", key="obesity_checkbox")
        abdominal_obesity = st.checkbox("Béo bụng", key="abdominal_obesity_checkbox")
        fat_tongue = st.checkbox("Lưỡi bệu", key="fat_tongue_checkbox")
        thick_coating = st.checkbox("Có rêu dày", key="thick_coating_checkbox")
        thin_coating = st.checkbox("Có rêu mỏng", key="thin_coating_checkbox")
        less_coating = st.checkbox("Ít rêu lưỡi", key="less_coating_checkbox")
        white_coating = st.checkbox("Rêu trắng", key="white_coating_checkbox")
        yellow_coating = st.checkbox("Rêu vàng", key="yellow_coating_checkbox")
        st.write("3. VĂN CHẨN", key="van_chan_section")
        fatigue = st.selectbox('Phạp lực', options=["không", "ít", "trung bình", "khá nhiều", "nhiều"], key="fatigue_checkbox")
        bad_breath = st.selectbox('Khí đoản', options=["không", "ít", "trung bình", "khá nhiều", "nhiều"], key="bad_breath_checkbox")

    with right:
        st.write("4. VẤN CHẨN")
        cold_limbs = st.checkbox("Chi lãnh", key="cold_limbs_checkbox")
        night_sweating = st.selectbox('Đạo hạn:', options=["không", "ít", "trung bình", "khá nhiều", "nhiều"], key="night_sweating_choice")
        vomiting = st.checkbox("Nôn ói", key="vomiting_checkbox")
        sticky_stool = st.checkbox("Phân dính", key="sticky_stool_checkbox")
        stinking_stool = st.checkbox("Phân hôi thối", key="stinking_stool_checkbox")
        blurred_vision = st.checkbox("Mục huyền", key="blurred_vision_checkbox")
        sore_weak_waist_knee = st.selectbox('Đau lưng mỏi gối:', options=["không", "ít", "trung bình", "khá nhiều", "nhiều"], key="sore_weak_waist_knee_choice")
        dreaminess = st.selectbox('Đa mị:', options=["không", "ít", "trung bình", "khá nhiều", "nhiều"], key="dreaminess_choice")
        st.write("5. THIẾT CHẨN")
        slippery_pulse = st.checkbox("Mạch hoạt",   key="slippery_pulse_checkbox")
        tight_pulse = st.checkbox("Mạch huyền",     key="tight_pulse_checkbox")
        rapid_pulse = st.checkbox("Mạch sác",     key="rapid_pulse_checkbox")

    submitted = st.form_submit_button("Run")

    if submitted:
        emaciation = 1 if emaciation else 0
        obesity = 1 if obesity else 0
        abdominal_obesity = 1 if abdominal_obesity else 0
        fat_tongue = 1 if fat_tongue else 0
        thick_coating = 1 if thick_coating else 0
        thin_coating = 1 if thin_coating else 0
        less_coating = 1 if less_coating else 0
        white_coating = 1 if white_coating else 0
        yellow_coating = 1 if yellow_coating else 0
        cold_limbs = 1 if cold_limbs else 0
        vomiting = 1 if vomiting else 0
        sticky_stool = 1 if sticky_stool else 0
        stinking_stool = 1 if stinking_stool else 0
        blurred_vision = 1 if blurred_vision else 0
        slippery_pulse = 1 if slippery_pulse else 0
        tight_pulse = 1 if tight_pulse else 0
        rapid_pulse = 1 if rapid_pulse else 0

        # Convert choices to numerical values
        mapping = {
            "không": 0,
            "ít": 1,
            "trung bình": 2,
            "khá nhiều": 3,
            "nhiều": 4
        }
        sore_weak_waist_knee_mapping = {
            "không": 0,
            "ít": 1,
            "trung bình": 2,
            "khá nhiều": 3,
            "nhiều": 4
        }
        dreaminess_mapping = {
            "không": 0,
            "ít": 1,
            "trung bình": 2,
            "khá nhiều": 3,
            "nhiều": 4
        }
        night_sweating = mapping.get(night_sweating, 0)
        sore_weak_waist_knee = mapping.get(sore_weak_waist_knee, 0)
        dreaminess = mapping.get(dreaminess, 0)
        fatigue = mapping.get(fatigue, 0)
        bad_breath = mapping.get(bad_breath, 0)
        # Prepare the input and run the prediction code here
        input_data = pd.DataFrame({
            "obesity": [obesity],
            "abdominal_obesity": [abdominal_obesity],
            "fat_tongue": [fat_tongue],
            "thick_coating": [thick_coating],
            "thin_coating": [thin_coating],
            "less_coating": [less_coating],
            "white_coating": [white_coating],
            "yellow_coating": [yellow_coating],
            "cold_limbs": [cold_limbs],
            "vomiting": [vomiting],
            "bmi": [bmi],
            "pulse_frequency": [pulse_frequency],
            "sticky_stool": [sticky_stool],
            "stinking_stool": [stinking_stool],
            "slippery_pulse": [slippery_pulse],
            "tight_pulse": [tight_pulse],
            "rapid_pulse": [rapid_pulse],
            "emaciation": [emaciation],
            "fatigue": [fatigue],
            "bad_breath": [bad_breath],            
            "night_sweating": [night_sweating],
            "sore_weak_waist_knee": [sore_weak_waist_knee],
            "blurred_vision": [blurred_vision],            
            "dreaminess": [dreaminess]
        })
        input_scaled = scaler.transform(input_data)
        y_pred = rf.predict(input_scaled)[0]
        y_proba = [float(val) for val in [arr[0, 1] for arr in rf.predict_proba(input_scaled)]]
### Result
st.markdown("========================================================================================")
st.markdown("<h2 style='text-align: center;'>Kết Quả</h2>", unsafe_allow_html=True)

label_names = ['Tỳ vị trở trệ', "Trường vị thấp nhiệt", "Khí âm lưỡng hư", "Can thận âm hư"]
output = {}
output_proba = {}
if submitted:
    for label, pred in zip(label_names, y_pred):
        output[label] = pred

    for label, proba in zip(label_names, y_proba):
        output_proba[label] = proba

    # CHẨN ĐOÁN
    st.title("Chẩn đoán:")
    if y_pred[0] == 1:
        st.write("Hội chứng Tỳ vị trở trệ")
    if y_pred[1] == 1:
        st.write("Hội chứng Trường vị thấp nhiệt")
    if y_pred[2] == 1:
        st.write("Hội chứng Khí âm lưỡng hư")
    if y_pred[3] == 1:
        st.write("Hội chứng Can thận âm hư")

    #Phương thang gợi ý
    st.title("Điều trị:")
    if y_pred[0] == 1:
        st.write("Hành khí đạo trệ - Hậu phác tam vật thang gia giảm (IV, ưu tiên thấp)")
    if y_pred[1] == 1:
        st.write("Thanh nhiệt, trừ thấp - Cát căn cầm liên thang (Ib, ưu tiên cao)")
    if y_pred[2] == 1:
        st.write("Bổ khí, dưỡng âm, thanh nhiệt - Sinh mạch tán (IIb, ưu tiên thấp)")
    if y_pred[3] == 1:
        st.write("Tư bổ Can Thận - Kỷ cúc địa hoàng hoàn (IIIa, ưu tiên thấp)")
        
    # Đảm bảo biểu đồ khép kín
    y_proba = np.append(y_proba, y_proba[0])
    angles = np.linspace(0, 2 * np.pi, len(label_names) + 1)

    # Vẽ radar chart với matplotlib
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(angles, y_proba, 'o-', linewidth=2)
    ax.fill(angles, y_proba, alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180/np.pi, label_names)
    ax.set_title("Radar chart các hội chứng")
    ax.grid(True)

    # Hiển thị lên Streamlit
    st.title("Radar chart xác suất các hội chứng")
    st.pyplot(fig)

st.markdown("========================================================================================")
st.markdown("<h2 style='text-align: left;'>Reference</h2>", unsafe_allow_html=True)
with st.expander("Bảng chẩn đoán các hội chứng"):
    st.table(output)

with st.expander("Bảng xác suất các hội chứng"):
    st.table(output_proba)

with st.expander("Thông tin người phát triển"):
    st.write("Mọi thắc mắc xin liên hệ tác giả")
    st.write("Bác sỹ: Nguyễn Lê Văn")
    st.write("Email: ylanhausinh1412@gmail.com")
    st.write("Mô hình được xây dựng dựa trên thuật toán Random Forest multilabel-Classification.")
    st.write("Công cụ thu thập: Bảng câu hỏi được soạn từ Guideline International traditional Chinese medicine guideline for diagnostic and treatment principles of diabetes (2020)")
    st.write("https://apm.amegroups.org/article/view/46744/html")
