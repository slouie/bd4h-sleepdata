import os
import numpy as np
import pyedflib
from urllib.request import urlretrieve


def load_edf_test():
    for record in ['ST7011J0-PSG', 'ST7011JP-Hypnogram']:
        url = "https://physionet.org/physiobank/database/sleep-edfx/sleep-telemetry/%s.edf" % record
        filename = "./data/%s.edf" % record
        urlretrieve(url, filename)
        reader = pyedflib.EdfReader(filename)

        print("\n======= %s =======\n" % record)
        print("edfsignals: %i" % reader.signals_in_file)
        print("file duration: %i seconds" % reader.file_duration)
        print("startdate: %i-%i-%i" % (reader.getStartdatetime().day,reader.getStartdatetime().month,reader.getStartdatetime().year))
        print("starttime: %i:%02i:%02i" % (reader.getStartdatetime().hour,reader.getStartdatetime().minute,reader.getStartdatetime().second))
        print("patientcode: %s" % reader.getPatientCode())
        print("gender: %s" % reader.getGender())
        print("birthdate: %s" % reader.getBirthdate())
        print("patient_name: %s" % reader.getPatientName())
        print("patient_additional: %s" % reader.getPatientAdditional())
        print("admincode: %s" % reader.getAdmincode())
        print("technician: %s" % reader.getTechnician())
        print("equipment: %s" % reader.getEquipment())
        print("recording_additional: %s" % reader.getRecordingAdditional())
        print("datarecord duration: %f seconds" % reader.getFileDuration())
        print("number of datarecords in the file: %i" % reader.datarecords_in_file)
        print("number of annotations in the file: %i\n" % reader.annotations_in_file)

        annotations = reader.readAnnotations()
        for n in np.arange(reader.annotations_in_file):
            print("annotation: onset is %f    duration is %s    description is %s" % (annotations[0][n],annotations[1][n],annotations[2][n]))

        for channel in range(reader.signals_in_file):
            print("signal parameters for the %d.channel:\n" % channel)

            print("label: %s" % reader.getLabel(channel))
            print("samples in file: %i" % reader.getNSamples()[channel])
            print("physical maximum: %f" % reader.getPhysicalMaximum(channel))
            print("physical minimum: %f" % reader.getPhysicalMinimum(channel))
            print("digital maximum: %i" % reader.getDigitalMaximum(channel))
            print("digital minimum: %i" % reader.getDigitalMinimum(channel))
            print("physical dimension: %s" % reader.getPhysicalDimension(channel))
            print("prefilter: %s" % reader.getPrefilter(channel))
            print("transducer: %s" % reader.getTransducer(channel))
            print("samplefrequency: %f\n" % reader.getSampleFrequency(channel))

            buf = reader.readSignal(channel)
            n = 200
            print("read %i samples\n" % n)
            result = ""
            for i in np.arange(n):
                result += ("%.1f, " % buf[i])
            print(result)
            print("\n")
        
        os.remove(filename)