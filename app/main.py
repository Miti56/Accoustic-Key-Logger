import record.record
import DataVisualisation.dataVisualisation
import DataVisualisation.modelVisualisation
import model.modelCC
import model.modelML
# import test.liveModel
def main():
    record.record.main()
    DataVisualisation.dataVisualisation.main()
    model.modelCC.main()
    model.modelML.main()
    DataVisualisation.modelVisualisation.main()
    # test.liveModel.main()



if __name__ == "__main__":
    main()