# import pyttsx3 as pytts
# def str2Voice(str):
#     engine=pytts.init()
#     engine.setProperty("rate",150)
#     engine.setProperty("volume",1)
#     voices=engine.getProperty("voices")
#     for voice in voices:
#         print('id={},name={}\n'.format(voice.id,voice.name))
#     engine.say(str)
#     engine.runAndWait()
#     engine.stop()
#
# if __name__=='__main__':
#     str2Voice("三角形")
import win32com.client as win
speaker=win.Dispatch("SAPI.SpVoice")
speaker.Speak("京A88888")