# intEnumを使用した方が良さそう
class ModelBasedConfig:

    modelbasedworkset_header = ('id',
                         'filename',
                         'subject_id',
                         'basedframe_id',
                         'picklepath')

    worksetid = 0
    filename = 1
    subject = 2
    baseframeid = 3
    picklepath = 4

    @classmethod
    def workset_define(cls, pickle_id: int, filename: str, path: str) -> tuple[int, str, str, str, str]:
        return (pickle_id, filename, '', '', path)