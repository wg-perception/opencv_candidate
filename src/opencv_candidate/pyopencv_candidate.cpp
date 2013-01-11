#include <Python.h>
#include <opencv_candidate/datamatrix.hpp>

PyObject *mod_opencv;

// These are sucky, sketchy versions of the real things in OpenCV Python,
// inferior in every way.

#define PYTHON_USE_NUMPY 0 // switch off for now...

static void translate_error_to_exception(void)
{
  PyErr_SetString(PyExc_RuntimeError, cvErrorStr(cvGetErrStatus()));
  cvSetErrStatus(0);
}
#define ERRCHK do { if (cvGetErrStatus() != 0) { translate_error_to_exception(); return NULL; } } while (0)
#define ERRWRAPN(F, N) \
    do { \
        try \
        { \
            F; \
        } \
        catch (const cv::Exception &e) \
        { \
           PyErr_SetString(PyExc_RuntimeError, e.err.c_str()); \
           return N; \
        } \
        ERRCHK; \
    } while(0)
#define ERRWRAP(F) ERRWRAPN(F, NULL) // for most functions, exception -> NULL return

struct cvmat_t {
  PyObject_HEAD
  CvMat *a;
  PyObject *data;
  size_t offset;
};

static int failmsg(const char *fmt, ...)
{
  char str[1000];

  va_list ap;
  va_start(ap, fmt);
  vsnprintf(str, sizeof(str), fmt, ap);
  va_end(ap);

  PyErr_SetString(PyExc_TypeError, str);
  return 0;
}

static int is_cvmat(PyObject *o)
{
  return 1;
}

static int convert_to_CvMat(PyObject *o, CvMat **dst, const char *name)
{
  cvmat_t *m = (cvmat_t*)o;
  void *buffer;
  Py_ssize_t buffer_len;

  if (!is_cvmat(o)) {
#if !PYTHON_USE_NUMPY
    return failmsg("Argument '%s' must be CvMat", name);
#else
    PyObject *asmat = fromarray(o, 0);
    if (asmat == NULL)
      return failmsg("Argument '%s' must be CvMat", name);
    // now have the array obect as a cvmat, can use regular conversion
    return convert_to_CvMat(asmat, dst, name);
#endif
  } else {
    m->a->refcount = NULL;
    if (m->data && PyString_Check(m->data)) {
      assert(cvGetErrStatus() == 0);
      char *ptr = PyString_AsString(m->data) + m->offset;
      cvSetData(m->a, ptr, m->a->step);
      assert(cvGetErrStatus() == 0);
      *dst = m->a;
      return 1;
    } else if (m->data && PyObject_AsWriteBuffer(m->data, &buffer, &buffer_len) == 0) {
      cvSetData(m->a, (void*)((char*)buffer + m->offset), m->a->step);
      assert(cvGetErrStatus() == 0);
      *dst = m->a;
      return 1;
    } else {
      return failmsg("CvMat argument '%s' has no data", name);
    }
  }
}

static PyObject *FROM_CvMat(CvMat *m)
{
  PyObject *creatematheader  = PyObject_GetAttrString(mod_opencv, "CreateMatHeader");
  PyObject *setdata  = PyObject_GetAttrString(mod_opencv, "SetData");
  PyObject *args;

  args = Py_BuildValue("iii", m->rows, m->cols, CV_MAT_TYPE(m->type));
  PyObject *pym = PyObject_CallObject(creatematheader, args);
  Py_DECREF(args);

  args = Py_BuildValue("Os#i", pym, m->data.ptr, m->rows * m->step, m->step);
  Py_DECREF(PyObject_CallObject(setdata, args));
  Py_DECREF(args);

  Py_DECREF(creatematheader);
  Py_DECREF(setdata);

  return pym;
}

static PyObject *pyfinddatamatrix(PyObject *self, PyObject *args)
{
  PyObject *pyim;
  if (!PyArg_ParseTuple(args, "O", &pyim))
    return NULL;

  CvMat *image = 0;
  if (!convert_to_CvMat(pyim, &image, "image")) return NULL;
  std::deque <CvDataMatrixCode> codes;
  cvFindDataMatrix(image);
  ERRWRAP(codes = cvFindDataMatrix(image));

  PyObject *pycodes = PyList_New(codes.size());
  size_t i;
  for (i = 0; i < codes.size(); i++) {
    CvDataMatrixCode *pc = &codes[i];
    PyList_SetItem(pycodes, i, Py_BuildValue("(sOO)", pc->msg, FROM_CvMat(pc->corners), FROM_CvMat(pc->original)));
  }

  return pycodes;
}

static PyMethodDef methods[] = {
  {"FindDataMatrix", pyfinddatamatrix, METH_VARARGS},
  {NULL, NULL},
};

extern "C"
void init_opencv_candidate()
{
  Py_InitModule("_opencv_candidate", methods);
  mod_opencv = PyImport_ImportModule("cv");
}
