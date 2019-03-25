init:
     pip install -r requirements.txt
test:
     py.test tests

.PHONY: all clean
.PHONY: init test
