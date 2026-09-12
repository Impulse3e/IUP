from agent.main import main
from agent.ui import report_crash

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as error:
        report_crash(error)
