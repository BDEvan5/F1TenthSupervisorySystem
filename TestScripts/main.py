from F1TenthSupervisorySystem.NavAgents.follow_the_gap import FollowTheGap
from F1TenthSupervisorySystem.NavAgents.PurePursuit import PurePursuit


from GeneralTrainTest import run_multi_test, test_vehicle, load_conf


config_file = "track_kernel"
# config_file = "config_test"

def test_pure_pursuit():
    conf = load_conf(config_file)
    vehicle = PurePursuit(conf)

    run_multi_test(conf, vehicle, 5)


def test_fgm():
    conf = load_conf(config_file)

    vehicle = FollowTheGap(conf)


    test_vehicle(conf, vehicle)
    # run_multi_test(conf, vehicle)



if __name__ == "__main__":
    test_pure_pursuit()
    # test_fgm()


