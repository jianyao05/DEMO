import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp
import time
import pose_module as pm  # Import your pose module
import threading





DEMO_VIDEO = 'properform2.mp4'
DEMO = True

# Setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

st.title('AI Squat Trainer')
st.sidebar.title('BioGait User Interface')

# Sidebar for App Modes
app_mode = st.sidebar.selectbox('Choose the App mode', ['About App', 'Articles and Research', 'Run on Video'])

if app_mode == 'About App':
    st.markdown(
        'This application uses **MediaPipe** for real-time pose detection and calculates squat angles to help users improve their exercise form. **Streamlit** is used for the GUI.')
    st.video('https://youtu.be/byxWus7BwfQ?si=b9zUkXcprE9moQjB&t=77')
    st.markdown('''
        # About BioGait \n
          BioGait is a company started by **Aloysius** and **Nadirah**. \n

          At **BioGait**, we seamlessly integrate artificial intelligence with healthcare to revolutionize physiotherapy and
           rehabilitation. Our advanced AI technology detects and analyzes key joint movements in **real-time**, offering 
           precise insights into your **form, technique, and alignment**. By leveraging human gait analysis, we ensure that 
           every exercise is performed **correctly and effectively**, helping you achieve **optimal recovery and improved 
           health outcomes**. With **BioGait**, you receive continuous, personalized support, making your rehabilitation 
           journey smoother and more successful. \n

        # Contact Us \n
          **ALOYSIUS TAN KIAT YAW**                               
          email: aloysiustankiatyaw@gmail.com                    
          contact number: 8808 9527 \n
          **NUR NADIRAH BINTE A JEFFERE**                             
          email: nurnadirah235@gmail.com                       
          contact number: 8776 4894


    ''')

elif app_mode == 'Articles and Research':
    st.markdown('''
            # **Articles and Research on Physiotherapy** \n
              **- 10 Tips for mental Recovery after a Sports Injury:**
              1. Acknowledge Your emotions 
              2. Set Realistic Goals \n
              [Read more to find out the rest of the tips...](https://www.newhopephysio.com/blog/10-tips-for-mental-recovery-after-a-sports-injury/)


              **- 10 Most Common Knee Injuries in Sports** \n
              Understand the various knee injuries to highlight **Preventive Measures and Rehabilitation strategies**. \n
              Learn the importance of proper warm-up and cool-down routines, strength training, flexibility exercises, and adopting appropriate techniques \n
              [Read more to find out the rest of the information...](https://www.newhopephysio.com/blog/10-most-common-knee-injuries-in-sports/)

              **- The Role of Physiotherapy in Sports Injury Rehabilitation** \n
              How help of **Physiotherapy**, athletes can regain their strength, mobility, and confidence to get back in the game \n
              Highlight the importance of Physiotherapy in having a Successful Recovery \n
              [Read more to find out the rest of the article...](https://www.newhopephysio.com/blog/role-of-physiotherapy-in-sports-injury-rehabilitation/)
        ''')


elif app_mode == 'Run on Video':
    # Configuration
    detection_confidence = st.sidebar.slider('Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    tracking_confidence = st.sidebar.slider('Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)

    limit = st.sidebar.slider('Full Squat Angle (Degrees)', min_value=20, max_value=160, value=80, step=1)

    use_webcam = st.sidebar.button('Use Webcam')
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    tffile = tempfile.NamedTemporaryFile(delete=False)

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown('**Frame Rate**')
        kpi1_text = st.markdown('0')

    with kpi2:
        st.markdown('**Amount of Proper Squat**')
        kpi2_text = st.markdown('0')

    with kpi3:
        st.markdown('**Range of Motion**')
        kpi3_text = st.markdown('0')

    st.markdown('<hr/>', unsafe_allow_html=True)

    if use_webcam or video_file_buffer or DEMO:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        elif video_file_buffer:
            tffile.write(video_file_buffer.read())
            vid = cv2.VideoCapture(tffile.name)
        elif DEMO:
            vid = cv2.VideoCapture(DEMO_VIDEO)
        else:
            tffile.write(video_file_buffer.read())
            vid = cv2.VideoCapture(tffile.name)

        # Pose Detector
        detector = pm.poseDetector(mode=False, smooth=True, detectionCon=detection_confidence,
                                   trackCon=tracking_confidence)

        # Create placeholders for live video and progress bars
        stframe = st.empty()
        progress_bar = st.progress(0)
        kpi1, kpi2, kpi3 = st.columns(3)

        count = 0
        dir = 0
        frame_count = 0
        angle = 0

        start_time = time.time()

        try:
            while True:
                ret, frame = vid.read()
                frame_count += 1
                if not ret:
                    break

                # Process frame
                frame = cv2.resize(frame, (1280, 720))
                frame = detector.findPose(frame, False)
                lmList = detector.findPosition(frame, False)

                if len(lmList) != 0:
                    angle = detector.findAngle(frame, 23, 25, 27)
                    per = np.interp(angle, (limit, 175), (100, 0))
                    progress_bar.progress(int(per))

                    # Color logic for bar
                    if per == 100:
                        if dir == 0:
                            count += 0.5
                            dir = 1
                            threading.Thread(target=play_audio).start()
                    if per == 0:
                        if dir == 1:
                            count += 0.5
                            dir = 0

                # Update KPIs
                kpi1_text.write(
                    f"<h1 style='text-align: center; color:red;'>{int(frame_count / (time.time() - start_time))}</h1>",
                    unsafe_allow_html=True)
                kpi2_text.write(f"<h1 style='text-align: center; color:red;'>{int(count)}</h1>", unsafe_allow_html=True)
                kpi3_text.write(f"<h1 style='text-align: center; color:red;'>{int(per)}%</h1>", unsafe_allow_html=True)

                # Display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame, use_column_width=True)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        finally:
            vid.release()

    else:
        st.write('Please select a video file or start webcam.')

