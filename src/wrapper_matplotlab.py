import matplotlib.pyplot as plt

fig0 = plt.figure()
ax0 = fig0.gca()
cf = ax0.contourf(RodThe, RodPhi, rod_u / (rod_W * lRod / 2))
fig0.colorbar(cf, ax=ax0)
ax0.set_title('u/(w*l)')
ax0.set_xlabel('theta')
ax0.set_ylabel('length')
fig0.savefig('%s_norm_u.png' % fileHeadle)
plt.close()

def