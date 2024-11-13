from astropy import table
from astropy.io import fits
from ..util.data import msg_format



class Filter(object):
    
    def __init__(self, event):
        
        msg = 'evt is not the type of astropy.tabel.Table'
        assert isinstance(event, table.Table), msg_format(msg)
        
        self._evt = event
        self.evt = self._evt.copy()
        
        self.exprs = []
        
    
    @classmethod
    def from_fits(cls, file, idx=None):
        
        hdu = fits.open(file)
        
        if idx is not None:
            evt = hdu[idx]
        else:
            try:
                evt = hdu['EVENTS']
            except KeyError:
                raise KeyError('EVENTS extension not found!')
            
        evt = table.Table.read(evt)
        hdu.close()

        return cls(evt)

    
    @property
    def tags(self):
        
        return self._evt.colnames
    
    
    @property
    def base(self):
        
        return {tag: self.evt[tag] for tag in self.tags}
    
    
    def info(self, tag=None):
        
        if tag is None:
            print(self.evt.info)
        else:
            msg = '%s is not one of tags' % tag
            assert tag in self.tags, msg_format(msg)
            
            print(self.evt[tag].info)
            
        print('\n'.join(self.exprs))


    def eval(self, expr):
        
        if expr is not None:
        
            flt = eval(expr, {}, self.base)
            self.evt = self.evt[flt]
            
            self.exprs.append(expr)
    
    
    def clear(self):
        
        self.evt = self._evt.copy()
        
        self.exprs = []
